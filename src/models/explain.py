"""Local tree-model attribution in predicted engine cycles."""
import numpy as np
import pandas as pd


def explain_prediction(model, row):
    import shap
    if len(row) != 1:
        raise ValueError('Select exactly one engine observation.')
    explainer = shap.TreeExplainer(model, feature_perturbation='tree_path_dependent')
    values = np.asarray(explainer.shap_values(row, check_additivity=True)).reshape(-1)
    baseline = float(np.asarray(explainer.expected_value).item())
    prediction = float(model.predict(row)[0])
    if len(values) != len(row.columns) or not np.isfinite(values).all():
        raise ValueError('Unsupported or invalid attribution output.')
    if not np.isclose(baseline + values.sum(), prediction, rtol=1e-5, atol=1e-5):
        raise ValueError('Attributions do not reconstruct the prediction.')
    table = pd.DataFrame({'Feature': row.columns, 'Reading': row.iloc[0].values,
                          'Contribution (cycles)': values})
    return baseline, prediction, table
