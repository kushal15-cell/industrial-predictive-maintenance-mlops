"""Preserve method-specific drift scores; never interpret distances as percentages."""
import math


def drift_details(result):
    rows = []
    for metric in result.get('metrics', []):
        config = metric.get('config', {})
        if 'ValueDrift' not in str(config.get('type', '')):
            continue
        score, threshold = metric.get('value'), config.get('threshold')
        method = str(config.get('method', 'Unknown'))
        valid = all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                    for v in (score, threshold))
        # Only known score semantics may determine drift. Unknown tests stay unknown.
        name = method.lower()
        direction = ('above' if any(s in name for s in ('wasserstein', 'jensen-shannon', 'population stability', 'psi'))
                     else 'below' if any(s in name for s in ('kolmogorov', 'ks', 'k-s', 'chi-square', 'fisher', 'z-test')) else None)
        drifted = (score > threshold if direction == 'above' else score < threshold) if valid and direction else None
        rows.append({'Feature': config.get('column'), 'Method': method,
                     'Score': score if valid else None, 'Threshold': threshold if valid else None,
                     'Drift when': direction or 'unknown',
                     'Status': 'Drift detected' if drifted else 'Within threshold' if drifted is False else 'Needs review'})
    return rows
