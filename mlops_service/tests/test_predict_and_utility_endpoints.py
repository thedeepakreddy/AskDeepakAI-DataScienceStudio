def test_predict_uses_stored_model(client, classification_rows):
    train_resp = client.post("/train", json={
        "data": classification_rows, "target": "Churn", "models": ["random_forest"]
    })
    model_id = train_resp.json()["champion"]["modelId"]

    predict_resp = client.post("/predict", json={"model_id": model_id, "features": classification_rows[:5]})
    assert predict_resp.status_code == 200
    predictions = predict_resp.json()["predictions"]
    assert len(predictions) == 5
    for p in predictions:
        assert p in ("Yes", "No")


def test_predict_unknown_model_id_returns_404(client):
    resp = client.post("/predict", json={"model_id": "does-not-exist", "features": [{"a": 1}]})
    assert resp.status_code == 404


def test_download_returns_real_joblib_bytes(client, classification_rows):
    train_resp = client.post("/train", json={
        "data": classification_rows, "target": "Churn", "models": ["random_forest"]
    })
    model_id = train_resp.json()["champion"]["modelId"]

    download_resp = client.get(f"/download/{model_id}")
    assert download_resp.status_code == 200
    assert len(download_resp.content) > 1000  # a real fitted pipeline isn't a tiny/empty file
    assert download_resp.headers["content-type"] == "application/octet-stream"


def test_download_unknown_model_id_returns_404(client):
    resp = client.get("/download/does-not-exist")
    assert resp.status_code == 404


def test_drift_metrics_detects_real_shift(client):
    reference = [{"value": 10 + i * 0.1} for i in range(100)]
    current = [{"value": 30 + i * 0.1} for i in range(100)]  # clearly shifted distribution

    resp = client.post("/drift-metrics", json={"reference_data": reference, "current_data": current})
    assert resp.status_code == 200
    drift = resp.json()["drift_status"]["value"]
    assert drift["drift_detected"] is True
    assert drift["p_value"] < 0.05


def test_drift_metrics_no_shift_for_identical_distributions(client):
    reference = [{"value": 10 + i * 0.1} for i in range(100)]
    current = [{"value": 10 + i * 0.1} for i in range(100)]

    resp = client.post("/drift-metrics", json={"reference_data": reference, "current_data": current})
    drift = resp.json()["drift_status"]["value"]
    assert drift["drift_detected"] is False


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
