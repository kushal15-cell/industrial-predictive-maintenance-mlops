import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from src.agent.decision_engine import evaluate_decision, get_baseline_metrics
from src.agent.retrain import promotion_gate, run, load_compatible_model
from src.agent.trace import new_report, now, event
from src.agent.approval import trigger_github_workflow
from src.agent.record_outcome import attach


def evidence():
    baseline = get_baseline_metrics()
    return {'generated_at': now(), 'sample_count': 150, 'drifted_feature_count': 2,
            'total_monitored_features': 25,
            'performance': {'mae': baseline['mae'] * 1.251, 'rmse': baseline['rmse'] * 1.251}}


class ReliabilityTests(unittest.TestCase):
    def test_incompatible_model_runtime_is_rejected(self):
        import warnings
        from sklearn.exceptions import InconsistentVersionWarning

        def incompatible_load(path):
            warnings.warn(InconsistentVersionWarning(
                estimator_name='RandomForestRegressor',
                current_sklearn_version='1.8.0', original_sklearn_version='1.9.0'))
            return object()

        with patch('joblib.load', side_effect=incompatible_load):
            with self.assertRaises(InconsistentVersionWarning):
                load_compatible_model('unused.pkl')

    def test_borderline_bad_recommendation_contained(self):
        self.assertTrue(evaluate_decision(evidence())['retraining_required'])
        result = promotion_gate({'mae': 10, 'rmse': 16}, {'mae': 11, 'rmse': 17})
        self.assertFalse(result['eligible'])

    def test_invalid_evidence_abstains(self):
        for change in ({'sample_count': 3}, {'generated_at': '2020-01-01T00:00:00+00:00'},
                       {'performance': {'mae': float('nan'), 'rmse': 20}},
                       {'drifted_feature_count': -1}):
            with self.subTest(change=change):
                self.assertEqual(evaluate_decision(evidence() | change)['decision'], 'WAIT_FOR_DATA')

    def test_gate_rejects_regression_ties_and_invalid_metrics(self):
        for candidate in ({'mae': 11, 'rmse': 15}, {'mae': 10, 'rmse': 16},
                          {'mae': 9, 'rmse': float('nan')}, {}):
            self.assertFalse(promotion_gate({'mae': 10, 'rmse': 16}, candidate)['eligible'])
        self.assertTrue(promotion_gate({'mae': 10, 'rmse': 16}, {'mae': 9, 'rmse': 15})['eligible'])

    def test_no_dispatch_without_approval(self):
        with patch('src.agent.approval.subprocess.run') as dispatch:
            with self.assertRaises(ValueError):
                trigger_github_workflow(new_report(evaluate_decision(evidence()), evidence()))
            dispatch.assert_not_called()

    def test_outcome_import_is_correlated_and_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = new_report(evaluate_decision(evidence()), evidence())
            with patch('src.agent.trace.trace_dir', return_value=root), patch('src.agent.record_outcome.trace_dir', return_value=root):
                event(report, 'DISPATCHED')
                (root / 'workflow.json').write_text(json.dumps({'decision_id': report['decision_id'],
                    'run_id': '123', 'run_attempt': '1', 'status': 'failure'}))
                result = attach(root)
                self.assertEqual(result['lifecycle'][-1]['status'], 'PIPELINE_FAILED')
                self.assertEqual(len(attach(root)['lifecycle']), 2)

    def test_nonfinite_evidence_can_be_recorded_for_review(self):
        with tempfile.TemporaryDirectory() as directory, patch('src.agent.trace.trace_dir', return_value=Path(directory)):
            summary = evidence() | {'performance': {'mae': float('nan'), 'rmse': 20}}
            report = new_report(evaluate_decision(summary), summary)
            event(report, 'REVIEWED')
            saved = json.loads(next(Path(directory).glob('*/record.json')).read_text())
            self.assertEqual(saved['decision']['decision'], 'WAIT_FOR_DATA')

    def test_trace_retains_multiple_decisions_and_events(self):
        with tempfile.TemporaryDirectory() as directory, patch('src.agent.trace.trace_dir', return_value=Path(directory)):
            for _ in range(2):
                report = new_report(evaluate_decision(evidence()), evidence())
                event(report, 'RECOMMENDED')
                event(report, 'REJECTED', actor='test')
            self.assertEqual(len(list(Path(directory).glob('*/record.json'))), 2)
            self.assertEqual(len(list(Path(directory).glob('*/events/*.json'))), 4)

    def test_training_failure_preserves_incumbent(self):
        import joblib
        import pandas as pd
        from sklearn.dummy import DummyRegressor
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'models').mkdir()
            (root / 'data/processed').mkdir(parents=True)
            (root / 'src').mkdir()
            (root / 'params.yaml').write_text('placeholder')
            data = pd.DataFrame({'unit_number': range(20), 'sensor': range(20), 'RUL': range(20)})
            data.to_csv(root / 'data/processed/train_fd001_processed.csv', index=False)
            model_path = root / 'models/random_forest_rul_model.pkl'
            joblib.dump(DummyRegressor().fit(data[['sensor']], data.RUL), model_path)
            original = model_path.read_bytes()
            def fail(*args, **kwargs):
                raise subprocess.CalledProcessError(1, args[0])
            with patch('src.agent.retrain.ROOT_DIR', root):
                result = run('test', root / 'output', trainer=fail)
            self.assertEqual(result['status'], 'PIPELINE_FAILED')
            self.assertEqual(original, model_path.read_bytes())
            self.assertFalse((root / 'output/candidate.pkl').exists())
            self.assertEqual(json.loads((root / 'output/outcome.json').read_text())['status'], 'PIPELINE_FAILED')

            def worse(*args, **kwargs):
                path = Path(kwargs['cwd']) / 'models/random_forest_rul_model.pkl'
                path.parent.mkdir()
                joblib.dump(DummyRegressor(strategy='constant', constant=100).fit(data[['sensor']], data.RUL), path)

            def better(*args, **kwargs):
                from sklearn.linear_model import LinearRegression
                path = Path(kwargs['cwd']) / 'models/random_forest_rul_model.pkl'
                path.parent.mkdir()
                # Injected perfect predictor tests artifact gating, not training quality.
                joblib.dump(LinearRegression().fit(data[['sensor']], data.RUL), path)

            with patch('src.agent.retrain.ROOT_DIR', root):
                blocked = run('test', root / 'blocked', trainer=worse)
                eligible = run('test', root / 'eligible', trainer=better)
                mismatch = run('test', root / 'mismatch', trainer=better, expected_incumbent='wrong')
            self.assertEqual(blocked['status'], 'PROMOTION_BLOCKED')
            self.assertFalse((root / 'blocked/candidate.pkl').exists())
            self.assertEqual(eligible['status'], 'CANDIDATE_ELIGIBLE')
            self.assertTrue((root / 'eligible/candidate.pkl').exists())
            self.assertEqual(mismatch['status'], 'PIPELINE_FAILED')
            self.assertEqual(original, model_path.read_bytes())


if __name__ == '__main__':
    unittest.main()
