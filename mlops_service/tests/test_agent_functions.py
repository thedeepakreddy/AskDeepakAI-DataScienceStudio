import random

import pytest


@pytest.fixture
def collinear_rows():
    """x1_copy is x1 plus tiny noise - genuinely, strongly collinear with x1,
    so a real VIF computation must flag it, unlike the independent features
    in regression_rows."""
    random.seed(11)
    rows = []
    for _ in range(200):
        x1 = random.uniform(0, 50)
        x1_copy = x1 + random.uniform(-0.05, 0.05)
        x2 = random.uniform(0, 20)
        y = 3 * x1 + 2 * x2 + 10 + random.uniform(-2, 2)
        rows.append({"x1": round(x1, 4), "x1_copy": round(x1_copy, 4), "x2": round(x2, 4), "target": round(y, 3)})
    return rows


def test_retrain_with_feature_transform_returns_real_metric(client, regression_rows):
    resp = client.post("/agent/retrain-with-transform", json={
        "data": regression_rows, "target": "target", "features": ["x1", "x2"],
        "model_key": "linear", "column": "x1", "method": "standardize",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["transformApplied"] == {"column": "x1", "method": "standardize"}
    assert body["targetWasTransformed"] is False
    # Standardizing a feature shouldn't change a linear model's real fit quality.
    assert body["primaryMetricValue"] > 0.9


def test_retrain_with_target_transform_flags_it(client, regression_rows):
    resp = client.post("/agent/retrain-with-transform", json={
        "data": regression_rows, "target": "target", "features": ["x1", "x2"],
        "model_key": "linear", "column": "target", "method": "log",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["targetWasTransformed"] is True
    assert 0.0 <= body["primaryMetricValue"] <= 1.0


def test_retrain_with_transform_unknown_model_key_returns_400(client, regression_rows):
    resp = client.post("/agent/retrain-with-transform", json={
        "data": regression_rows, "target": "target", "features": ["x1", "x2"],
        "model_key": "not_a_real_model", "column": "x1", "method": "log",
    })
    assert resp.status_code == 400


def test_retrain_with_transform_non_numeric_column_returns_400(client, classification_rows):
    resp = client.post("/agent/retrain-with-transform", json={
        "data": classification_rows, "target": "Churn", "features": ["tenure", "MonthlyCharges", "ContractType"],
        "model_key": "random_forest", "column": "ContractType", "method": "log",
    })
    assert resp.status_code == 400


def test_drop_feature_retrains_without_it(client, regression_rows):
    rows = [{**r, "irrelevant": random.random()} for r in regression_rows]
    resp = client.post("/agent/drop-feature", json={
        "data": rows, "target": "target", "features": ["x1", "x2", "irrelevant"],
        "model_key": "linear", "drop_feature": "irrelevant",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["droppedFeature"] == "irrelevant"
    assert body["remainingFeatures"] == ["x1", "x2"]
    assert body["primaryMetricValue"] > 0.9


def test_drop_feature_cannot_drop_the_only_feature(client, regression_rows):
    resp = client.post("/agent/drop-feature", json={
        "data": regression_rows, "target": "target", "features": ["x1"],
        "model_key": "linear", "drop_feature": "x1",
    })
    assert resp.status_code == 400


def test_vif_flags_genuinely_collinear_feature(client, collinear_rows):
    resp = client.post("/agent/check-multicollinearity", json={
        "data": collinear_rows, "features": ["x1", "x1_copy", "x2"],
    })
    assert resp.status_code == 200
    body = resp.json()
    # x1_copy ~= x1 + tiny noise, so R^2 regressing one on the others can be
    # near-perfect - the endpoint honestly reports vif=None (undefined/near-
    # infinite) rather than a fabricated bounded number in that case, so this
    # only asserts on the real, always-present highRisk flag.
    assert "x1" in body["highRiskFeatures"]
    assert "x1_copy" in body["highRiskFeatures"]
    assert "x2" not in body["highRiskFeatures"]


def test_vif_does_not_flag_independent_features(client, regression_rows):
    resp = client.post("/agent/check-multicollinearity", json={
        "data": regression_rows, "features": ["x1", "x2"],
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["highRiskFeatures"] == []


def test_vif_too_few_numeric_features_returns_empty(client, regression_rows):
    resp = client.post("/agent/check-multicollinearity", json={
        "data": regression_rows, "features": ["x1"],
    })
    assert resp.status_code == 200
    assert resp.json()["vifScores"] == []
