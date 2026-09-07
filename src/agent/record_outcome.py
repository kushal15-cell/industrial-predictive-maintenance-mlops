"""Attach a downloaded CI audit artifact to its local decision history."""
import argparse
import json
from pathlib import Path
from src.agent.trace import event, trace_dir


def attach(directory):
    directory = Path(directory)
    workflow = json.loads((directory / 'workflow.json').read_text(encoding='utf-8'))
    decision_id = workflow['decision_id']
    if not isinstance(decision_id, str) or len(decision_id) != 32 or any(c not in '0123456789abcdef' for c in decision_id):
        raise ValueError('Artifact does not reference a local decision ID.')
    path = trace_dir() / decision_id / 'record.json'
    report = json.loads(path.read_text(encoding='utf-8'))
    if any(e.get('run_id') == workflow['run_id'] and e.get('run_attempt') == workflow['run_attempt'] for e in report['lifecycle']):
        return report
    outcome_path = directory / 'outcome.json'
    outcome = json.loads(outcome_path.read_text(encoding='utf-8')) if outcome_path.exists() else {}
    if outcome and outcome.get('decision_id') != decision_id:
        raise ValueError('Workflow and evaluation decision IDs differ.')
    status = outcome.get('status', 'PIPELINE_FAILED')
    if workflow['status'] != 'success':
        status = 'PIPELINE_FAILED'
    event(report, status, run_id=workflow['run_id'], run_attempt=workflow['run_attempt'],
          workflow=workflow, evaluation=outcome)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('artifact_directory')
    print(attach(parser.parse_args().artifact_directory)['lifecycle'][-1]['status'])
