"""Validate a reviewed, DVC-versioned candidate before requesting deployment."""
import argparse
import json
from pathlib import Path
from src.agent.retrain import promotion_gate, load_compatible_model
from src.agent.trace import digest
from src.config import ROOT_DIR


def validate_release(path):
    record = json.loads(Path(path).read_text(encoding='utf-8'))
    if not record.get('approved_by') or record.get('approved') is not True:
        raise ValueError('Explicit deployment approval is required')
    outcome = record['evaluation']
    if outcome['status'] != 'CANDIDATE_ELIGIBLE' or not promotion_gate(
            outcome['gate']['incumbent'], outcome['gate']['candidate'])['eligible']:
        raise ValueError('Candidate did not pass evaluation')
    model_path = ROOT_DIR / 'models/random_forest_rul_model.pkl'
    if not outcome.get('candidate_sha256') or digest(model_path) != outcome['candidate_sha256']:
        raise ValueError('DVC-restored model differs from approved candidate')
    load_compatible_model(model_path)
    return outcome['candidate_sha256']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest')
    print(validate_release(parser.parse_args().manifest))
