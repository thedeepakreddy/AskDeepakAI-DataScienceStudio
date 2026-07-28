def test_train_classification_response_shape(client, classification_rows):
    resp = client.post("/train", json={"data": classification_rows, "target": "Churn"})
    assert resp.status_code == 200
    body = resp.json()

    assert body["task"] == "classification"
    assert body["isBinary"] is True
    assert "champion" in body and "comparison" in body and "results" in body
    assert isinstance(body["selectionReason"], str) and len(body["selectionReason"]) > 0

    # AutoML must try multiple real candidates, not just train one model.
    assert len(body["results"]) >= 3
    assert len(body["comparison"]) == len(body["results"])

    champion = body["champion"]
    metrics = champion["metrics"]
    for key in ("accuracy", "precision", "recall", "f1Score"):
        assert 0.0 <= metrics[key] <= 1.0

    # This fixture has a strong, learnable signal — a real model should beat
    # random guessing by a wide margin.
    assert metrics["accuracy"] > 0.7

    cm = champion["confusionMatrix"]
    assert cm["tp"] + cm["tn"] + cm["fp"] + cm["fn"] == champion["testRows"]

    assert champion["cv"]["folds"] == 5
    assert 0.0 <= champion["cv"]["mean"] <= 1.0

    feature_names = {f["feature"] for f in champion["featureImportance"]}
    assert feature_names.issubset({"tenure", "MonthlyCharges", "ContractType"})


def test_train_regression_response_shape(client, regression_rows):
    resp = client.post("/train", json={
        "data": regression_rows, "target": "target", "models": ["linear", "random_forest"]
    })
    assert resp.status_code == 200
    body = resp.json()

    assert body["task"] == "regression"
    metrics = body["champion"]["metrics"]
    assert "r2Score" in metrics and "mae" in metrics and "rmse" in metrics
    assert metrics["rmse"] >= 0 and metrics["mae"] >= 0

    # y = 3*x1 + 2*x2 + 10 is almost perfectly linear — a real fit should score high.
    assert metrics["r2Score"] > 0.85


def test_train_ranks_by_cross_validation(client, classification_rows):
    """The core assertion behind Phase 1's selection change: comparison must
    actually be ordered by CV mean, not by the single test-set score."""
    resp = client.post("/train", json={"data": classification_rows, "target": "Churn"})
    comparison = resp.json()["comparison"]
    cv_means = [c["cv"]["mean"] for c in comparison if c.get("cv") and not c["cv"].get("error")]
    assert cv_means == sorted(cv_means, reverse=True)


def test_train_missing_target_column_returns_400(client, classification_rows):
    resp = client.post("/train", json={"data": classification_rows, "target": "DoesNotExist"})
    assert resp.status_code == 400
    assert "detail" in resp.json()


def test_train_too_few_rows_returns_400(client):
    resp = client.post("/train", json={"data": [{"a": 1, "b": "x"}] * 5, "target": "b"})
    assert resp.status_code == 400


def test_train_single_class_target_returns_400(client):
    rows = [{"a": i, "b": "SameValue"} for i in range(30)]
    resp = client.post("/train", json={"data": rows, "target": "b"})
    assert resp.status_code == 400


def test_train_shap_importance_present_for_champion(client, classification_rows):
    resp = client.post("/train", json={
        "data": classification_rows, "target": "Churn", "models": ["random_forest"]
    })
    shap_importance = resp.json()["champion"]["shapImportance"]
    assert shap_importance is not None and len(shap_importance) > 0
    for item in shap_importance:
        assert "feature" in item and "score" in item and item["score"] >= 0
