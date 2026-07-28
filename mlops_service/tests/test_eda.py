import pandas as pd


def test_eda_response_shape(client, classification_rows):
    resp = client.post("/eda", json={"data": classification_rows})
    assert resp.status_code == 200
    body = resp.json()

    assert body["rowCount"] == len(classification_rows)
    assert body["columnCount"] == 4
    for key in ("numericSummaries", "outliers", "correlation", "missingReport", "categoricalSummaries"):
        assert key in body

    assert {s["column"] for s in body["numericSummaries"]} == {"tenure", "MonthlyCharges"}
    assert {s["column"] for s in body["categoricalSummaries"]} == {"ContractType", "Churn"}


def test_eda_numeric_stats_match_independent_pandas_computation(client, classification_rows):
    """The real assertion: /eda's numbers must match what pandas itself
    computes directly over the same data, not just be present and shaped right."""
    resp = client.post("/eda", json={"data": classification_rows})
    body = resp.json()

    df = pd.DataFrame(classification_rows)
    for summary in body["numericSummaries"]:
        series = df[summary["column"]].dropna()
        assert summary["mean"] == round(float(series.mean()), 4)
        assert summary["median"] == round(float(series.median()), 4)
        assert summary["min"] == round(float(series.min()), 4)
        assert summary["max"] == round(float(series.max()), 4)
        assert summary["skew"] == round(float(series.skew()), 4)


def test_eda_missing_report_accuracy(client, eda_rows):
    resp = client.post("/eda", json={"data": eda_rows})
    missing = next(m for m in resp.json()["missingReport"] if m["column"] == "normal_col")
    assert missing["missingCount"] == 5


def test_eda_outlier_detection_finds_injected_outliers(client, eda_rows):
    resp = client.post("/eda", json={"data": eda_rows})
    outliers = next(o for o in resp.json()["outliers"] if o["column"] == "normal_col")
    # 3 extreme values (500, 520, 540) were injected far outside the [10, 20] range.
    assert outliers["outlierCount"] >= 3


def test_eda_correlation_matrix_is_symmetric_and_bounded(client, classification_rows):
    resp = client.post("/eda", json={"data": classification_rows})
    matrix = resp.json()["correlation"]["matrix"]
    n = len(matrix)
    for i in range(n):
        assert matrix[i][i] == 1.0
        for j in range(n):
            assert matrix[i][j] == matrix[j][i]
            assert -1.0 <= matrix[i][j] <= 1.0


def test_eda_empty_data_returns_400(client):
    resp = client.post("/eda", json={"data": []})
    assert resp.status_code == 400


def test_eda_single_row_returns_400(client):
    resp = client.post("/eda", json={"data": [{"a": 1}]})
    assert resp.status_code == 400
