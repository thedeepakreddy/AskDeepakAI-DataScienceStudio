def test_numeric_vs_numeric_uses_pearson_correlation(client, regression_rows):
    resp = client.post("/hypothesis-test", json={"data": regression_rows, "columns": ["x1", "target"]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["testName"] == "Pearson Correlation"
    # x1 is a strong real driver of target (y = 3*x1 + 2*x2 + noise) - the
    # correlation must genuinely be strong and significant, not just present.
    assert body["testStatistic"] > 0.5
    assert body["pValue"] < 0.001
    assert body["rejectNull"] is True


def test_numeric_vs_binary_categorical_uses_welch_ttest(client, classification_rows):
    resp = client.post("/hypothesis-test", json={"data": classification_rows, "columns": ["tenure", "Churn"]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["testName"] == "Two-Sample T-Test (Welch)"
    # tenure only interacts with Churn jointly with ContractType in this fixture
    # (see test_categorical_vs_categorical below for the strong marginal signal),
    # so this just checks the mechanics produce a valid, genuine p-value.
    assert 0.0 <= body["pValue"] <= 1.0
    assert body["sampleSize"] == len(classification_rows)


def test_numeric_vs_three_group_categorical_uses_anova(client, classification_rows):
    resp = client.post("/hypothesis-test", json={"data": classification_rows, "columns": ["tenure", "ContractType"]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["testName"] == "One-Way ANOVA"
    assert "sampleSize" in body


def test_categorical_vs_categorical_uses_chi_squared(client, classification_rows):
    resp = client.post("/hypothesis-test", json={"data": classification_rows, "columns": ["ContractType", "Churn"]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["testName"] == "Chi-Squared Test of Independence"
    # Real injected signal: churn concentrates in Month-to-month contracts.
    assert body["pValue"] < 0.01
    assert body["rejectNull"] is True


def test_two_independent_random_columns_do_not_reject_null(client):
    """Sanity check the test isn't just always significant: two genuinely
    unrelated columns should NOT show a significant relationship."""
    import random
    random.seed(99)
    rows = [{"a": random.uniform(0, 1), "b": random.uniform(0, 1)} for _ in range(200)]
    resp = client.post("/hypothesis-test", json={"data": rows, "columns": ["a", "b"]})
    body = resp.json()
    assert body["pValue"] > 0.05
    assert body["rejectNull"] is False


def test_wrong_number_of_columns_returns_400(client, classification_rows):
    resp = client.post("/hypothesis-test", json={"data": classification_rows, "columns": ["tenure"]})
    assert resp.status_code == 400


def test_missing_column_returns_400(client, classification_rows):
    resp = client.post("/hypothesis-test", json={"data": classification_rows, "columns": ["tenure", "NotAColumn"]})
    assert resp.status_code == 400


def test_empty_data_returns_400(client):
    resp = client.post("/hypothesis-test", json={"data": [], "columns": ["a", "b"]})
    assert resp.status_code == 400


def test_too_few_complete_rows_returns_400(client):
    resp = client.post("/hypothesis-test", json={"data": [{"a": 1, "b": 2}], "columns": ["a", "b"]})
    assert resp.status_code == 400
