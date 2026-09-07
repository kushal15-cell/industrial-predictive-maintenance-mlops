import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from fastapi import HTTPException
from sklearn.ensemble import RandomForestRegressor

from src.api.main import app, lifespan, health, predict, PredictionRequest
from src.models.explain import explain_prediction
from src.monitoring.drift_details import drift_details
from src.agent.release import validate_release


class InterpretabilityTests(unittest.TestCase):
    def setUp(self):
        self.x = pd.DataFrame({'sensor_1': [1., 2., 3., 4.], 'sensor_2': [4., 3., 2., 1.]})
        self.model = RandomForestRegressor(n_estimators=3, random_state=42).fit(self.x, [10., 20., 30., 40.])

    def test_attributions_reconstruct_actual_prediction(self):
        base, prediction, table = explain_prediction(self.model, self.x.iloc[[0]])
        self.assertAlmostEqual(base + table['Contribution (cycles)'].sum(), prediction)
        self.assertEqual(list(table.Feature), list(self.x.columns))

    def test_drift_score_direction_is_method_specific(self):
        def metric(method, value):
            return {'config': {'type': 'ValueDrift', 'column': 'sensor_1', 'method': method, 'threshold': .1}, 'value': value}
        result = drift_details({'metrics': [metric('Wasserstein distance (normed)', .2),
                                            metric('K-S p_value', .01), metric('unknown', .2)]})
        self.assertEqual([r['Status'] for r in result], ['Drift detected', 'Drift detected', 'Needs review'])

    def test_api_readiness_and_schema_validation(self):
        async def check():
            with patch('src.api.main.load_compatible_model', return_value=self.model):
                async with lifespan(app):
                    self.assertEqual(health()['status'], 'ready')
                    result = predict(PredictionRequest(features={'sensor_1': 1., 'sensor_2': 4.}))
                    self.assertIsNone(result['prediction_interval'])
                    with self.assertRaises(HTTPException) as caught:
                        predict(PredictionRequest(features={'wrong_sensor': 1.}))
                    self.assertEqual(caught.exception.status_code, 422)
            with patch('src.api.main.load_compatible_model', side_effect=ValueError('incompatible')):
                async with lifespan(app):
                    with self.assertRaises(HTTPException) as caught:
                        health()
                    self.assertEqual(caught.exception.status_code, 503)
        asyncio.run(check())

    def test_release_rejects_unapproved_or_wrong_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'release.json'
            path.write_text(json.dumps({'approved': False}))
            with self.assertRaises(ValueError):
                validate_release(path)
            record = {'approved': True, 'approved_by': 'test reviewer', 'evaluation': {
                'status': 'CANDIDATE_ELIGIBLE', 'candidate_sha256': 'expected',
                'gate': {'incumbent': {'mae': 10, 'rmse': 16}, 'candidate': {'mae': 9, 'rmse': 15}}}}
            path.write_text(json.dumps(record))
            with patch('src.agent.release.digest', return_value='different'):
                with self.assertRaises(ValueError):
                    validate_release(path)
            with patch('src.agent.release.digest', return_value='expected'), patch('src.agent.release.load_compatible_model'):
                self.assertEqual(validate_release(path), 'expected')


if __name__ == '__main__':
    unittest.main()
