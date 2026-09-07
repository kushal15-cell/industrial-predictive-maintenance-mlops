"""Reproducible injected scenario; no API calls or deployment side effects."""
from src.agent.decision_engine import evaluate_decision, get_baseline_metrics
from src.agent.retrain import promotion_gate
from src.agent.trace import new_report, now, write_json
from src.config import ROOT_DIR


def main():
    baseline = get_baseline_metrics()
    summary = {'generated_at': now(), 'sample_count': 150,
               'window': {'kind': 'synthetic_batch', 'start': None, 'end': None},
               'drifted_feature_count': 2, 'total_monitored_features': 25,
               'performance': {key: baseline[key] * 1.251 for key in ('mae', 'rmse')}}
    report = new_report(evaluate_decision(summary), summary)
    report['scenario'] = 'INJECTED TEST: borderline recommendation, worse candidate'
    report['lifecycle'] = [
        {'timestamp': now(), 'status': 'RECOMMENDED'},
        {'timestamp': now(), 'status': 'APPROVED', 'actor': 'simulated reviewer'},
        {'timestamp': now(), 'status': 'PROMOTION_BLOCKED', 'promotion': 'NOT_PROMOTED',
         'gate': promotion_gate({'mae': 10, 'rmse': 16}, {'mae': 11, 'rmse': 17})}]
    path = ROOT_DIR / 'docs/examples/borderline-decision.json'
    write_json(path, report)
    print(path)


if __name__ == '__main__':
    main()
