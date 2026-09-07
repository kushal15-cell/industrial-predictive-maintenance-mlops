"""Train in an isolated directory; publish only candidates that pass evaluation."""
import argparse
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import warnings
from importlib.metadata import version

from src.agent.trace import digest, now, write_json
from src.config import ROOT_DIR, load_params


def load_compatible_model(path):
    """Do not evaluate a pickle under an incompatible sklearn runtime."""
    import joblib
    from sklearn.exceptions import InconsistentVersionWarning
    with warnings.catch_warnings():
        warnings.simplefilter('error', InconsistentVersionWarning)
        return joblib.load(path)


def promotion_gate(incumbent, candidate):
    for metrics in (incumbent, candidate):
        for key in ('mae', 'rmse'):
            value = metrics.get(key)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                return {'eligible': False, 'reason': 'Missing or invalid evaluation metrics.'}
    eligible = candidate['rmse'] < incumbent['rmse'] and candidate['mae'] <= incumbent['mae']
    return {'eligible': eligible, 'reason': 'Lower RMSE and no MAE regression required.',
            'incumbent': incumbent, 'candidate': candidate}


def run(decision_id, output, trainer=None, expected_incumbent=None):
    output = Path(output)
    # Never reuse a directory containing a candidate from an earlier attempt.
    output.mkdir(parents=True, exist_ok=False)
    record = {'decision_id': decision_id, 'started_at': now(), 'status': 'RUNNING',
              'run_id': os.getenv('GITHUB_RUN_ID'), 'run_attempt': os.getenv('GITHUB_RUN_ATTEMPT'),
              'commit': os.getenv('GITHUB_SHA'), 'promotion': 'NOT_PROMOTED'}
    write_json(output / 'outcome.json', record)
    try:
        import joblib
        import pandas as pd
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        from sklearn.model_selection import train_test_split

        params = load_params()
        incumbent_path = ROOT_DIR / 'models/random_forest_rul_model.pkl'
        data_path = ROOT_DIR / params['data']['processed_path'] / 'train_fd001_processed.csv'
        record['incumbent_sha256'] = digest(incumbent_path)
        if expected_incumbent and record['incumbent_sha256'] != expected_incumbent:
            raise ValueError('Restored incumbent differs from the approved decision model.')
        record['dataset_sha256'] = digest(data_path)
        record['runtime'] = {'python': sys.version.split()[0],
                             **{name: version(name) for name in ('scikit-learn', 'numpy', 'joblib', 'pandas')}}
        incumbent = load_compatible_model(incumbent_path)
        df = pd.read_csv(data_path)
        _, test_units = train_test_split(df.unit_number.unique(),
            test_size=params['model']['test_size'], random_state=params['model']['random_state'])
        heldout = df[df.unit_number.isin(test_units)]
        record['evaluation'] = {'rows': len(heldout), 'engine_ids': test_units.tolist(),
                                'policy': 'same frozen engine holdout for both models'}
        X = heldout.drop(columns=['unit_number', 'RUL'])
        y = heldout.RUL

        def metrics(model):
            predictions = model.predict(X)
            return {'mae': float(mean_absolute_error(y, predictions)),
                    'rmse': float(mean_squared_error(y, predictions) ** 0.5)}

        baseline = metrics(incumbent)
        with tempfile.TemporaryDirectory(prefix='ipmip-candidate-') as directory:
            workspace = Path(directory)
            shutil.copytree(ROOT_DIR / 'src', workspace / 'src', ignore=shutil.ignore_patterns('__pycache__'))
            shutil.copy2(ROOT_DIR / 'params.yaml', workspace / 'params.yaml')
            target = workspace / params['data']['processed_path'] / data_path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(data_path, target)
            with (output / 'training.log').open('w', encoding='utf-8') as log:
                (trainer or subprocess.run)([sys.executable, '-m', 'src.models.train_model'],
                    cwd=workspace, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=3600)
            candidate_path = workspace / 'models/random_forest_rul_model.pkl'
            candidate = metrics(load_compatible_model(candidate_path))
            record['candidate_sha256'] = digest(candidate_path)
            record['gate'] = promotion_gate(baseline, candidate)
            record['status'] = 'CANDIDATE_ELIGIBLE' if record['gate']['eligible'] else 'PROMOTION_BLOCKED'
            if record['gate']['eligible']:
                shutil.copy2(candidate_path, output / 'candidate.pkl')
    except Exception as error:
        record['status'] = 'PIPELINE_FAILED'
        record['error'] = f'{type(error).__name__}: {error}'
    finally:
        record['finished_at'] = now()
        write_json(output / 'outcome.json', record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--decision-id', required=True)
    parser.add_argument('--output', default='retraining-result')
    parser.add_argument('--expected-incumbent')
    args = parser.parse_args()
    result = run(args.decision_id, args.output, expected_incumbent=args.expected_incumbent)
    print(result['status'])
    sys.exit(1 if result['status'] == 'PIPELINE_FAILED' else 0)
