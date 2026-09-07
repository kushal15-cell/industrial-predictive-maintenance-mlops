"""Reviewable evidence snapshots and append-only lifecycle events."""
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from src.config import ROOT_DIR, load_params


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    path = Path(path)
    if not path.is_file():
        return None
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def trace_dir():
    return ROOT_DIR / load_params()['monitoring']['report_dir'] / 'decisions'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {'invalid_numeric_value': str(value)}
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def event(report, status, **details):
    entry = {'timestamp': now(), 'status': status, **details}
    report.setdefault('lifecycle', []).append(entry)
    # Separate immutable event files avoid losing history when the latest view changes.
    path = trace_dir() / report['decision_id'] / 'events' / (uuid4().hex + '.json')
    write_json(path, entry)
    write_json(trace_dir() / report['decision_id'] / 'record.json', report)
    return entry


def new_report(decision, summary=None):
    return {
        'schema_version': 1,
        'decision_id': uuid4().hex,
        'timestamp': now(),
        'agent': 'industrial_predictive_maintenance_agent',
        'engine': 'threshold_rules_v1',
        'model_sha256': digest(ROOT_DIR / 'models/random_forest_rul_model.pkl'),
        'policy': load_params()['agent'],
        'monitoring_snapshot': summary,
        'decision': decision,
        'lifecycle': [],
    }
