"""
AskDeepakAI MLOps Service
--------------------------
Real, honest model training. Every number returned by this service comes from
an actually-fitted scikit-learn / XGBoost estimator evaluated on a held-out
test split — nothing here is narrated or guessed by an LLM.
"""
import gc
import os
import time
import uuid
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
import scipy.stats as stats
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:  # pragma: no cover - environment-dependent
    HAS_XGBOOST = False

try:
    import shap
    HAS_SHAP = True
except ImportError:  # pragma: no cover - environment-dependent
    HAS_SHAP = False


class LabelEncodedClassifier(BaseEstimator, ClassifierMixin):
    """Drop-in classifier wrapper for estimators (like XGBoost) that require
    integer-encoded targets and reject raw labels such as 'Yes'/'No'. Encoding
    and decoding happen transparently, so this behaves like any other fitted
    sklearn classifier to the rest of the pipeline: .classes_ is exposed in the
    *original* label space, .predict() returns original labels, and
    .predict_proba() columns line up with .classes_ order.

    Inherits BaseEstimator/ClassifierMixin (rather than being a plain class) so
    it remains a well-behaved estimator under sklearn's clone()/tag machinery,
    which Pipeline and cross_val_score both rely on."""

    def __init__(self, base_estimator=None):
        self.base_estimator = base_estimator

    def fit(self, X, y):
        self._encoder = LabelEncoder()
        y_encoded = self._encoder.fit_transform(y)
        self.base_estimator.fit(X, y_encoded)
        self.classes_ = self._encoder.classes_
        return self

    def predict(self, X):
        encoded_pred = self.base_estimator.predict(X)
        return self._encoder.inverse_transform(encoded_pred.astype(int))

    def predict_proba(self, X):
        return self.base_estimator.predict_proba(X)

    @property
    def feature_importances_(self):
        return self.base_estimator.feature_importances_

app = FastAPI(title="AskDeepakAI MLOps Service")

MAX_STORED_MODELS = 25
model_store: dict[str, dict[str, Any]] = {}
model_order: list[str] = []

VALID_MODEL_KEYS = ["linear", "random_forest", "gradient_boosting", "xgboost", "mlp"]
# AutoML mode trains all 5 real candidates and ranks them — the point of an
# AutoML pipeline is comparing genuine alternatives, not training one model.
DEFAULT_MODELS = ["linear", "random_forest", "gradient_boosting", "xgboost", "mlp"]


# --------------------------------------------------------------------------
# Request/response schemas
# --------------------------------------------------------------------------

class TrainRequest(BaseModel):
    data: list[dict[str, Any]]
    target: str
    features: list[str] | None = None
    task_type: Literal["classification", "regression"] | None = None
    models: list[str] | None = None
    test_size: float = Field(default=0.2, gt=0.0, lt=0.9)
    cv_folds: int = Field(default=5, ge=2, le=10)
    random_state: int = 42
    hyperparameters: dict[str, Any] | None = None


class PredictRequest(BaseModel):
    model_id: str
    features: list[dict[str, Any]]


class DriftRequest(BaseModel):
    reference_data: list[dict[str, Any]]
    current_data: list[dict[str, Any]]


class EdaRequest(BaseModel):
    data: list[dict[str, Any]]


class HypothesisTestRequest(BaseModel):
    data: list[dict[str, Any]]
    columns: list[str]


class AgentRetrainTransformRequest(BaseModel):
    data: list[dict[str, Any]]
    target: str
    features: list[str] | None = None
    model_key: str
    column: str
    method: Literal["log", "sqrt", "standardize"]
    task_type: Literal["classification", "regression"] | None = None


class AgentDropFeatureRequest(BaseModel):
    data: list[dict[str, Any]]
    target: str
    features: list[str] | None = None
    model_key: str
    drop_feature: str
    task_type: Literal["classification", "regression"] | None = None


class AgentVifRequest(BaseModel):
    data: list[dict[str, Any]]
    features: list[str]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def build_preprocessor(X: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_estimator(model_key: str, task: str, hyperparameters: dict[str, Any]):
    n_estimators = int(hyperparameters.get("n_estimators", 100))
    max_depth = hyperparameters.get("max_depth")
    max_depth = int(max_depth) if max_depth not in (None, "") else None
    learning_rate = float(hyperparameters.get("learning_rate", 0.1))
    epochs = int(hyperparameters.get("epochs", 200))
    random_state = int(hyperparameters.get("random_state", 42))

    if model_key == "linear":
        return (LogisticRegression(max_iter=1000, random_state=random_state)
                if task == "classification"
                else LinearRegression())

    if model_key == "random_forest":
        cls = RandomForestClassifier if task == "classification" else RandomForestRegressor
        # oob_score=True gives a real, free out-of-bag performance estimate
        # (each tree is evaluated only on the rows it didn't see during bagging)
        # without costing an extra train/test split.
        return cls(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, oob_score=True)

    if model_key == "gradient_boosting":
        cls = GradientBoostingClassifier if task == "classification" else GradientBoostingRegressor
        return cls(n_estimators=n_estimators, max_depth=max_depth or 3, learning_rate=learning_rate, random_state=random_state)

    if model_key == "xgboost":
        if not HAS_XGBOOST:
            # Honest fallback: tell the caller, don't silently substitute without saying so.
            raise ValueError("xgboost is not installed on this service")
        if task == "classification":
            # XGBoost requires integer-encoded targets (0..n-1) and rejects raw
            # string labels like "Yes"/"No" — wrap it so it accepts the same
            # label space as every other classifier in this service.
            return LabelEncodedClassifier(XGBClassifier(
                n_estimators=n_estimators, max_depth=max_depth or 6,
                learning_rate=learning_rate, random_state=random_state,
                eval_metric="logloss"
            ))
        return XGBRegressor(
            n_estimators=n_estimators, max_depth=max_depth or 6,
            learning_rate=learning_rate, random_state=random_state
        )

    if model_key == "mlp":
        hidden = (64, 32)
        cls = MLPClassifier if task == "classification" else MLPRegressor
        # early_stopping=True holds out 10% of training data as a real internal
        # validation set, giving genuine validation_scores_ per epoch (not just
        # a training loss curve) and stopping before max_iter if it plateaus.
        return cls(hidden_layer_sizes=hidden, max_iter=epochs, random_state=random_state, early_stopping=True, n_iter_no_change=10)

    raise ValueError(f"Unknown model key: {model_key}")


def display_name(model_key: str, task: str) -> str:
    names = {
        "linear": "Logistic Regression" if task == "classification" else "Linear Regression",
        "random_forest": "Random Forest " + ("Classifier" if task == "classification" else "Regressor"),
        "gradient_boosting": "Gradient Boosting " + ("Classifier" if task == "classification" else "Regressor"),
        "xgboost": "XGBoost " + ("Classifier" if task == "classification" else "Regressor"),
        "mlp": "Multi-Layer Perceptron (Neural Network)",
    }
    return names.get(model_key, model_key)


def get_transformed_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return []


def aggregate_importance_to_original(raw_names: list[str], raw_scores: list[float]) -> list[dict[str, Any]]:
    """One-hot encoding explodes a single original column into many transformed
    columns (e.g. num__Age, cat__Country_US, cat__Country_UK). Sum the
    contributions back onto the original column name so the UI shows a
    feature importance chart users can actually map back to their data."""
    agg: dict[str, float] = {}
    for name, score in zip(raw_names, raw_scores):
        # strip the ColumnTransformer prefix ("num__" / "cat__")
        stripped = name.split("__", 1)[-1] if "__" in name else name
        # strip the one-hot encoded category suffix, e.g. "Country_US" -> "Country"
        original = stripped
        if "_" in stripped:
            # only strip if this looks like a OHE expansion (best-effort heuristic)
            candidate = stripped.rsplit("_", 1)[0]
            original = candidate if candidate else stripped
        agg[original] = agg.get(original, 0.0) + abs(float(score))

    total = sum(agg.values()) or 1.0
    result = [{"feature": k, "score": round(v / total, 4)} for k, v in agg.items()]
    result.sort(key=lambda r: r["score"], reverse=True)
    return result


def extract_feature_importance(estimator, preprocessor: ColumnTransformer) -> list[dict[str, Any]]:
    names = get_transformed_feature_names(preprocessor)
    if hasattr(estimator, "feature_importances_"):
        scores = list(estimator.feature_importances_)
    elif hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_)
        scores = list(np.abs(coef).ravel()) if coef.ndim > 1 else list(np.abs(coef))
    else:
        return []
    if not names or len(names) != len(scores):
        return [{"feature": f"feature_{i}", "score": round(float(s), 4)} for i, s in enumerate(scores)]
    return aggregate_importance_to_original(names, scores)


def extract_rf_tree_splits(estimator, preprocessor: ColumnTransformer, max_trees: int = 6) -> list[dict[str, Any]]:
    """Pull genuine root-split data out of real fitted DecisionTree estimators
    inside a RandomForest. Nothing here is invented — it's read directly off
    sklearn's tree_.feature / tree_.threshold / tree_.impurity arrays."""
    if not hasattr(estimator, "estimators_"):
        return []
    names = get_transformed_feature_names(preprocessor)
    trees = estimator.estimators_[:max_trees]
    out = []
    for idx, tree_model in enumerate(trees):
        t = tree_model.tree_
        feat_idx = int(t.feature[0])
        raw_name = names[feat_idx] if names and 0 <= feat_idx < len(names) else f"feature_{feat_idx}"
        feat_name = raw_name.split("__", 1)[-1] if "__" in raw_name else raw_name
        out.append({
            "id": idx + 1,
            "name": f"Estimator Tree #{idx + 1}",
            "splitFeature": feat_name,
            "splitValue": round(float(t.threshold[0]), 4),
            "sampleCount": int(t.n_node_samples[0]),
            "impurity": round(float(t.impurity[0]), 4),
            "treeDepth": int(t.max_depth),
            "leafCount": int(t.n_leaves),
        })
    return out


def choose_positive_label(y_train, classes_: list[Any]):
    """For binary targets with arbitrary label values (e.g. 'Yes'/'No',
    'Active'/'Churned'), there's no universal 'positive' class. We deterministically
    treat the minority class in the training data as positive — the standard
    convention for problems like churn, fraud, or default prediction, where the
    class of interest is the rarer one."""
    counts = pd.Series(y_train).value_counts()
    pos_label = counts.idxmin()
    neg_label = [c for c in classes_ if c != pos_label][0]
    pos_idx = list(classes_).index(pos_label)
    return pos_label, neg_label, pos_idx


def compute_classification_metrics(y_test, y_pred, y_proba, is_binary: bool, pos_label=None, pos_idx=None) -> dict[str, Any]:
    if is_binary:
        metrics: dict[str, Any] = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred, pos_label=pos_label, average="binary", zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, pos_label=pos_label, average="binary", zero_division=0)), 4),
            "f1Score": round(float(f1_score(y_test, y_pred, pos_label=pos_label, average="binary", zero_division=0)), 4),
        }
        if y_proba is not None and pos_idx is not None:
            try:
                metrics["rocAuc"] = round(float(roc_auc_score(y_test, y_proba[:, pos_idx], pos_label=pos_label)), 4)
            except Exception:
                pass
        return metrics

    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "f1Score": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
    }


def compute_confusion_matrix(y_test, y_pred, is_binary: bool, pos_label=None, neg_label=None):
    if is_binary and pos_label is not None:
        labels = [neg_label, pos_label]
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        tn, fp, fn, tp = cm.ravel()
        return {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn), "positiveLabel": str(pos_label)}

    labels = sorted(pd.unique(pd.Series(list(y_test) + list(y_pred))), key=str)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    return {"matrix": cm.tolist(), "labels": [str(lbl) for lbl in labels]}


def compute_regression_metrics(y_test, y_pred) -> dict[str, Any]:
    mse = mean_squared_error(y_test, y_pred)
    return {
        "r2Score": round(float(r2_score(y_test, y_pred)), 4),
        "mse": round(float(mse), 4),
        "rmse": round(float(np.sqrt(mse)), 4),
        "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
    }


def to_native(value):
    """Convert numpy scalars (int64, float64, bool_, str_...) to plain Python
    types so FastAPI's JSON encoder doesn't choke on them, and NaN -> None."""
    if isinstance(value, (np.generic,)):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def apply_column_transform(series: pd.Series, method: str) -> pd.Series:
    """Real, deterministic column transforms for the agentic loop. Shifts
    only when the data's actual range requires it to stay in-domain, so a
    log transform on already-positive data is untouched."""
    s = series.astype(float)
    if method == "log":
        min_val = float(s.min())
        shift = (1.0 - min_val) if min_val <= 0 else 0.0
        return np.log1p(s + shift)
    if method == "sqrt":
        min_val = float(s.min())
        shift = (0.0 - min_val) if min_val < 0 else 0.0
        return np.sqrt(s + shift)
    if method == "standardize":
        std = float(s.std())
        return (s - s.mean()) / std if std > 0 else s - s.mean()
    raise ValueError(f"Unknown transform method: {method}")


def compute_vif(df: pd.DataFrame, numeric_cols: list[str]) -> list[dict[str, Any]]:
    """Real Variance Inflation Factor per numeric feature: VIF_i = 1 / (1 - R^2_i),
    where R^2_i comes from actually regressing feature i on every other
    numeric feature. VIF > 5 is the conventional multicollinearity flag."""
    sub = df[numeric_cols].dropna()
    results = []
    if len(sub) < 5 or len(numeric_cols) < 2:
        return results
    for col in numeric_cols:
        other_cols = [c for c in numeric_cols if c != col]
        y = sub[col].to_numpy()
        X_other = sub[other_cols].to_numpy()
        reg = LinearRegression().fit(X_other, y)
        r_squared = reg.score(X_other, y)
        vif = None if r_squared >= 0.9999 else round(1.0 / (1.0 - r_squared), 3)
        # vif=None means "effectively infinite" (near-perfect collinearity),
        # which is the worst case, not an exemption from the high-risk flag.
        results.append({
            "feature": col,
            "vif": vif,
            "rSquared": round(float(r_squared), 4),
            "highRisk": bool(vif is None or vif > 5),
        })
    return results


def rank_by_cross_validation(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """AutoML model selection rule, made explicit rather than hidden: rank by
    mean cross-validation score (more robust than a single held-out test
    split), highest first; ties broken by lowest fold-to-fold standard
    deviation (the more consistent model wins). Models whose CV run failed
    are ranked after every model with valid CV, ordered among themselves by
    their held-out test score as a fallback."""
    def sort_key(r):
        cv = r.get("cv") or {}
        cv_mean = cv.get("mean")
        cv_std = cv.get("std")
        if cv_mean is None:
            return (1, 0.0, -r["primaryMetricValue"])
        return (0, -cv_mean, cv_std if cv_std is not None else 0.0)

    return sorted(results, key=sort_key)


def build_selection_reason(results: list[dict[str, Any]]) -> str:
    """Explains the AutoML pick in plain language, using only the real
    numbers already computed above — this is the exact rule rank_by_cross_validation
    applies, just spelled out for the UI instead of left implicit."""
    champion = results[0]
    cv = champion.get("cv") or {}
    cv_mean = cv.get("mean")

    if cv_mean is None:
        return (f"{champion['modelName']} selected on held-out test {champion['primaryMetric']} "
                f"({champion['primaryMetricValue']}) — cross-validation could not be computed for this run.")

    reason = (f"{champion['modelName']} selected for the highest {cv.get('folds')}-fold cross-validation "
              f"mean {cv.get('metric')} ({round(cv_mean, 4)})")

    if len(results) > 1:
        runner_up = results[1]
        ru_cv = runner_up.get("cv") or {}
        ru_mean = ru_cv.get("mean")
        if ru_mean is not None and abs(cv_mean - ru_mean) < 0.005:
            champ_std = cv.get("std", 0.0)
            ru_std = ru_cv.get("std", 0.0)
            reason += (f", tie-broken against {runner_up['modelName']} (CV mean {round(ru_mean, 4)}) "
                       f"by lower fold-to-fold variance (std {round(champ_std, 4)} vs {round(ru_std, 4)})")

    return reason + "."


def compute_shap_importance(pipeline: Pipeline, X_sample: pd.DataFrame, model_key: str, task: str, max_features: int = 10) -> list[dict[str, Any]] | None:
    """Computes real SHAP values for the champion model on a bounded sample
    of the held-out test set, then aggregates mean(|SHAP value|) back to
    original column names. Returns None (never fabricated numbers) if SHAP
    can't explain this particular model/version combination."""
    if not HAS_SHAP:
        return None
    try:
        preprocessor = pipeline.named_steps["preprocess"]
        estimator = pipeline.named_steps["model"]
        X_transformed = preprocessor.transform(X_sample)
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()
        names = get_transformed_feature_names(preprocessor)

        # LabelEncodedClassifier (used for xgboost classification) wraps the
        # real estimator SHAP needs to introspect directly.
        shap_estimator = estimator.base_estimator if isinstance(estimator, LabelEncodedClassifier) else estimator

        if model_key in ("random_forest", "gradient_boosting", "xgboost"):
            explainer = shap.TreeExplainer(shap_estimator)
            raw_values = explainer.shap_values(X_transformed)
        elif model_key == "linear":
            explainer = shap.LinearExplainer(shap_estimator, X_transformed)
            raw_values = explainer.shap_values(X_transformed)
        else:
            # Generic bounded explainer (e.g. MLP) — background sample kept
            # small since this path is far more compute-expensive. For
            # classification this must explain predict_proba (numeric), not
            # predict (returns string class labels SHAP's masking can't
            # handle numerically).
            background_n = min(30, len(X_transformed))
            predict_fn = (
                shap_estimator.predict_proba
                if task == "classification" and hasattr(shap_estimator, "predict_proba")
                else shap_estimator.predict
            )
            explainer = shap.Explainer(predict_fn, X_transformed[:background_n])
            raw_values = explainer(X_transformed).values

        # Normalize the many shapes SHAP can return across versions/model
        # families into a single (n_samples, n_features) magnitude array.
        if isinstance(raw_values, list):
            arr = np.mean([np.abs(np.asarray(v)) for v in raw_values], axis=0)
        else:
            arr = np.asarray(raw_values)
            if arr.ndim == 3:
                arr = np.abs(arr).mean(axis=-1)
            else:
                arr = np.abs(arr)

        mean_abs = np.asarray(arr).mean(axis=0).flatten()
        if not names or len(names) != len(mean_abs):
            return None

        importance = aggregate_importance_to_original(names, list(mean_abs))
        return importance[:max_features]
    except Exception:
        return None


def store_model(pipeline: Pipeline, meta: dict[str, Any]) -> str:
    model_id = str(uuid.uuid4())
    model_store[model_id] = {"pipeline": pipeline, "meta": meta}
    model_order.append(model_id)
    if len(model_order) > MAX_STORED_MODELS:
        oldest = model_order.pop(0)
        model_store.pop(oldest, None)
    return model_id


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "xgboost_available": HAS_XGBOOST}


@app.post("/train")
def train_model(request: TrainRequest):
    if not request.data:
        raise HTTPException(status_code=400, detail="No training data provided.")
    if len(request.data) < 20:
        raise HTTPException(status_code=400, detail=f"Need at least 20 rows to train reliably; received {len(request.data)}.")

    df = pd.DataFrame(request.data)
    if request.target not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{request.target}' not found in the dataset.")

    feature_cols = request.features or [c for c in df.columns if c != request.target]
    feature_cols = [c for c in feature_cols if c in df.columns and c != request.target]
    if not feature_cols:
        raise HTTPException(status_code=400, detail="No valid feature columns provided.")

    y_raw = df[request.target]
    X = df[feature_cols].copy()

    # Drop rows where the target is missing — can't train or evaluate on unknown labels.
    valid_mask = y_raw.notna()
    X = X[valid_mask]
    y_raw = y_raw[valid_mask]
    if len(X) < 20:
        raise HTTPException(status_code=400, detail="Fewer than 20 rows remain after dropping missing target values.")

    # Determine task type
    numeric_target = pd.api.types.is_numeric_dtype(y_raw)
    distinct = y_raw.nunique()
    if request.task_type:
        task = request.task_type
    else:
        task = "regression" if (numeric_target and distinct > 15) else "classification"

    if task == "classification" and distinct < 2:
        raise HTTPException(status_code=400, detail="Target column has fewer than 2 distinct classes; cannot classify.")

    y = y_raw
    is_binary = task == "classification" and distinct == 2

    requested_models = [m for m in (request.models or DEFAULT_MODELS) if m in VALID_MODEL_KEYS]
    if not requested_models:
        requested_models = DEFAULT_MODELS

    # Single split shared across all requested models for a fair, apples-to-apples comparison.
    stratify = y if (task == "classification" and y.value_counts().min() >= 2) else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=request.test_size, random_state=request.random_state, stratify=stratify
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=request.test_size, random_state=request.random_state
        )

    hyperparameters = request.hyperparameters or {}
    scoring = "accuracy" if task == "classification" else "r2"
    cv_folds = min(request.cv_folds, y_train.value_counts().min()) if task == "classification" else request.cv_folds
    cv_folds = max(cv_folds, 2)
    # Large datasets: cap folds at 3. More rows per fold already gives a
    # statistically stable CV estimate as data grows, so the dominant cost
    # (a full pipeline refit per fold, per candidate model) stays roughly
    # bounded instead of scaling with both row count and fold count at once -
    # this is what keeps AutoML mode (5 candidates x refits) from running the
    # free-tier host out of memory/time on a real-sized dataset.
    if len(X_train) > 5000:
        cv_folds = min(cv_folds, 3)

    results = []
    errors = []
    # Every candidate's fitted pipeline is kept here only for the duration of
    # this request. Only the champion actually gets handed to store_model()
    # below - the other candidates were only ever needed for this comparison,
    # and permanently caching all 5 in the global model_store on every
    # AutoML run was pure memory bloat that accumulates across a session.
    fitted_pipelines: dict[str, Pipeline] = {}

    for model_key in requested_models:
        try:
            preprocessor = build_preprocessor(X_train, scale_numeric=(model_key in ("linear", "mlp")))
            estimator = make_estimator(model_key, task, hyperparameters)
            pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])

            start = time.time()
            pipeline.fit(X_train, y_train)
            elapsed_ms = int((time.time() - start) * 1000)

            y_pred = pipeline.predict(X_test)

            # Cross-validation on the training fold only (never touches the held-out test set).
            try:
                cv_splitter = (StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=request.random_state)
                               if task == "classification"
                               else KFold(n_splits=cv_folds, shuffle=True, random_state=request.random_state))
                cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv_splitter, scoring=scoring)
                cv_result = {
                    "scores": [round(float(s), 4) for s in cv_scores],
                    "mean": round(float(np.mean(cv_scores)), 4),
                    "std": round(float(np.std(cv_scores)), 4),
                    "folds": cv_folds,
                    "metric": scoring,
                }
            except Exception as cv_err:
                cv_result = {"error": str(cv_err)}

            fitted_preprocessor = pipeline.named_steps["preprocess"]
            fitted_estimator = pipeline.named_steps["model"]

            if task == "classification":
                y_proba = pipeline.predict_proba(X_test) if hasattr(fitted_estimator, "predict_proba") else None
                pos_label = neg_label = pos_idx = None
                if is_binary and hasattr(fitted_estimator, "classes_"):
                    pos_label, neg_label, pos_idx = choose_positive_label(y_train, fitted_estimator.classes_)
                metrics = compute_classification_metrics(y_test, y_pred, y_proba, is_binary, pos_label, pos_idx)
                confusion = compute_confusion_matrix(y_test, y_pred, is_binary, pos_label, neg_label)
                primary_metric_name = "accuracy"
                primary_metric_value = metrics["accuracy"]
            else:
                metrics = compute_regression_metrics(y_test, y_pred)
                confusion = None
                primary_metric_name = "r2Score"
                primary_metric_value = metrics["r2Score"]

            feature_importance = extract_feature_importance(fitted_estimator, fitted_preprocessor)

            estimators_detail = (
                extract_rf_tree_splits(fitted_estimator, fitted_preprocessor)
                if model_key == "random_forest" else []
            )

            oob_score = None
            if model_key == "random_forest" and hasattr(fitted_estimator, "oob_score_"):
                try:
                    oob_score = round(float(fitted_estimator.oob_score_), 4)
                except Exception:
                    oob_score = None

            deep_learning = None
            if model_key == "mlp" and hasattr(fitted_estimator, "loss_curve_"):
                validation_scores = getattr(fitted_estimator, "validation_scores_", None)
                deep_learning = {
                    "trainingLogs": [
                        {"epoch": i + 1, "trainingLoss": round(float(loss), 5)}
                        for i, loss in enumerate(fitted_estimator.loss_curve_)
                    ],
                    "validationScores": (
                        [round(float(v), 5) for v in validation_scores]
                        if validation_scores is not None else None
                    ),
                    "validationMetric": "accuracy" if task == "classification" else "r2",
                    "finalValidationScore": (
                        round(float(validation_scores[-1]), 5)
                        if validation_scores is not None and len(validation_scores) > 0 else None
                    ),
                    "hiddenLayerSizes": list(fitted_estimator.hidden_layer_sizes),
                    "nLayers": int(fitted_estimator.n_layers_),
                    "nIterations": int(fitted_estimator.n_iter_),
                    "totalTrainableParams": sum(
                        w.size for w in fitted_estimator.coefs_
                    ) + sum(b.size for b in fitted_estimator.intercepts_),
                }

            # Real actual-vs-predicted rows for the frontend prediction table.
            # y_pred is positionally aligned with X_test (same row order), so we
            # index both by position `i`, not by the original dataframe index.
            sample_n = min(50, len(X_test))
            predictions = []
            for i in range(sample_n):
                row_idx = X_test.index[i]
                confidence = None
                if task == "classification" and y_proba is not None:
                    confidence = round(float(np.max(y_proba[i])) * 100, 1)
                predictions.append({
                    "id": i + 1,
                    "actual": to_native(y_test.iloc[i]),
                    "predicted": to_native(y_pred[i]),
                    "confidence": confidence,
                    "features": {c: to_native(X_test.loc[row_idx, c]) for c in feature_cols[:5]},
                })

            results.append({
                "modelKey": model_key,
                "modelName": display_name(model_key, task),
                "modelId": None,  # only the selected champion gets stored - see below
                "metrics": metrics,
                "confusionMatrix": confusion,
                "featureImportance": feature_importance,
                "cv": cv_result,
                "estimators": estimators_detail,
                "oobScore": oob_score,
                "deepLearning": deep_learning,
                "predictions": predictions,
                "executionTimeMs": elapsed_ms,
                "primaryMetric": primary_metric_name,
                "primaryMetricValue": primary_metric_value,
                "trainRows": int(len(X_train)),
                "testRows": int(len(X_test)),
            })
            fitted_pipelines[model_key] = pipeline
        except Exception as model_err:
            errors.append({"modelKey": model_key, "error": str(model_err)})
        finally:
            # Encourage prompt release of the (potentially large) pipeline,
            # preprocessed arrays, and CV fold copies for this candidate
            # before moving on to the next one - AutoML mode fits 5 of these
            # in a row, and peak memory on a constrained host is dominated by
            # how much of that stays garbage-but-uncollected at once.
            gc.collect()

    if not results:
        raise HTTPException(status_code=500, detail=f"All requested models failed to train: {errors}")

    results = rank_by_cross_validation(results)
    champion = results[0]
    selection_reason = build_selection_reason(results)

    # Only the champion's fitted pipeline is ever referenced again (by
    # /predict and /download), so it's the only one actually persisted.
    champion_pipeline = fitted_pipelines[champion["modelKey"]]
    champion["modelId"] = store_model(champion_pipeline, {
        "task": task, "target": request.target, "features": feature_cols, "modelKey": champion["modelKey"],
    })

    # Real SHAP explainability for the champion only (computing it for every
    # candidate would multiply training time for no benefit to the user).
    champion["shapImportance"] = None
    if HAS_SHAP:
        sample_n = min(100, len(X_test))
        X_shap_sample = X_test.iloc[:sample_n]
        champion["shapImportance"] = compute_shap_importance(
            champion_pipeline, X_shap_sample, champion["modelKey"], task
        )

    fitted_pipelines.clear()
    gc.collect()

    return {
        "status": "success",
        "task": task,
        "isBinary": is_binary,
        "target": request.target,
        "features": feature_cols,
        "champion": champion,
        "selectionReason": selection_reason,
        "comparison": [
            {
                "modelKey": r["modelKey"],
                "modelName": r["modelName"],
                "primaryMetric": r["primaryMetric"],
                "metricValue": r["primaryMetricValue"],
                "executionTimeMs": r["executionTimeMs"],
                "cv": r.get("cv"),
            } for r in results
        ],
        "results": results,
        "errors": errors,
        "trainRatio": round(1 - request.test_size, 2),
    }


@app.post("/eda")
def run_eda(request: EdaRequest):
    """Real, server-side exploratory data analysis. Everything returned here
    is computed directly by pandas/numpy over the actual uploaded rows — mean,
    median, std, skew, IQR-based outliers, and Pearson correlation are all
    genuine statistics, not narrated guesses."""
    if not request.data:
        raise HTTPException(status_code=400, detail="No data provided.")

    df = pd.DataFrame(request.data)
    if len(df) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 rows to compute EDA statistics.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    missing_report = [
        {
            "column": col,
            "missingCount": int(df[col].isna().sum()),
            "missingPercent": round(float(df[col].isna().sum()) / len(df) * 100, 2),
        }
        for col in df.columns
    ]

    numeric_summaries = []
    outlier_report = []
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        numeric_summaries.append({
            "column": col,
            "count": int(len(series)),
            "mean": round(float(series.mean()), 4),
            "median": round(float(series.median()), 4),
            "std": round(float(series.std(ddof=0)), 4) if len(series) > 1 else 0.0,
            "min": round(float(series.min()), 4),
            "max": round(float(series.max()), 4),
            "q1": round(q1, 4),
            "q3": round(q3, 4),
            # Fisher-Pearson skewness: 0 symmetric, >0 right-tailed, <0 left-tailed.
            "skew": round(float(series.skew()), 4) if len(series) > 2 else 0.0,
        })

        outlier_mask = (series < lower_bound) | (series > upper_bound)
        outlier_series = series[outlier_mask]
        outlier_report.append({
            "column": col,
            "q1": round(q1, 4),
            "q3": round(q3, 4),
            "iqr": round(iqr, 4),
            "lowerBound": round(lower_bound, 4),
            "upperBound": round(upper_bound, 4),
            "outlierCount": int(outlier_mask.sum()),
            "outlierPercent": round(float(outlier_mask.sum()) / len(series) * 100, 2),
            "sampleOutliers": [
                {"rowIndex": to_native(idx), "value": to_native(val)}
                for idx, val in outlier_series.head(50).items()
            ],
        })

    correlation = None
    if len(numeric_cols) >= 2:
        corr_df = df[numeric_cols].corr(method="pearson")
        correlation = {
            "columns": numeric_cols,
            "matrix": [
                [None if pd.isna(v) else round(float(v), 4) for v in row]
                for row in corr_df.values.tolist()
            ],
        }

    categorical_summaries = []
    for col in categorical_cols:
        non_null_total = int(df[col].notna().sum())
        top_counts = df[col].value_counts(dropna=True).head(10)
        categorical_summaries.append({
            "column": col,
            "distinctCount": int(df[col].nunique(dropna=True)),
            "topValues": [
                {
                    "value": str(idx),
                    "count": int(cnt),
                    "percent": round(int(cnt) / non_null_total * 100, 2) if non_null_total > 0 else 0.0,
                }
                for idx, cnt in top_counts.items()
            ],
        })

    return {
        "rowCount": int(len(df)),
        "columnCount": int(len(df.columns)),
        "numericSummaries": numeric_summaries,
        "outliers": outlier_report,
        "correlation": correlation,
        "missingReport": missing_report,
        "categoricalSummaries": categorical_summaries,
    }


@app.post("/predict")
def predict_model(request: PredictRequest):
    stored = model_store.get(request.model_id)
    if not stored:
        raise HTTPException(status_code=404, detail="Model not found. It may have expired — train again.")

    df = pd.DataFrame(request.features)
    pipeline: Pipeline = stored["pipeline"]
    try:
        predictions = pipeline.predict(df)
    except Exception as err:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {err}")

    return {"predictions": predictions.tolist(), "meta": stored["meta"]}


@app.get("/download/{model_id}")
def download_model(model_id: str, background_tasks: BackgroundTasks):
    stored = model_store.get(model_id)
    if not stored:
        raise HTTPException(status_code=404, detail="Model not found. It may have expired — train again.")

    filename = f"/tmp/{model_id}.joblib"
    joblib.dump(stored["pipeline"], filename)
    background_tasks.add_task(lambda: os.path.exists(filename) and os.remove(filename))
    download_name = f"{stored['meta'].get('target', 'model')}_{stored['meta'].get('modelKey', 'model')}.joblib"
    return FileResponse(filename, filename=download_name, media_type="application/octet-stream")


@app.post("/drift-metrics")
def drift_metrics(request: DriftRequest):
    """Computes real data drift using the Kolmogorov-Smirnov two-sample test,
    per numeric feature, between a reference slice and a current slice."""
    if not request.current_data or not request.reference_data:
        raise HTTPException(status_code=400, detail="Both reference_data and current_data are required.")

    df_ref = pd.DataFrame(request.reference_data)
    df_cur = pd.DataFrame(request.current_data)

    drift_report = {}
    for col in df_cur.columns:
        if col in df_ref.columns and pd.api.types.is_numeric_dtype(df_ref[col]):
            ref_vals = df_ref[col].dropna()
            cur_vals = df_cur[col].dropna()
            if len(ref_vals) < 2 or len(cur_vals) < 2:
                continue
            stat, p_value = stats.ks_2samp(ref_vals, cur_vals)
            drift_report[col] = {
                "ks_stat": round(float(stat), 4),
                "p_value": round(float(p_value), 4),
                "drift_detected": bool(p_value < 0.05),
            }

    return {"drift_status": drift_report}


@app.post("/hypothesis-test")
def run_hypothesis_test(request: HypothesisTestRequest):
    """Runs a genuine statistical test between two real columns from the
    actual dataset. The test type - Pearson correlation, Welch's t-test,
    one-way ANOVA, or a chi-squared test of independence - is chosen
    automatically from each column's real dtype and group cardinality, and
    computed via scipy.stats. Never a placeholder or random value."""
    if not request.data:
        raise HTTPException(status_code=400, detail="No data provided.")
    if len(request.columns) != 2:
        raise HTTPException(status_code=400, detail="Exactly two columns are required to test a relationship.")

    df = pd.DataFrame(request.data)
    col_a, col_b = request.columns
    missing = [c for c in (col_a, col_b) if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Column(s) not found in data: {missing}")

    sub = df[[col_a, col_b]].dropna()
    if len(sub) < 4:
        raise HTTPException(status_code=400, detail="Need at least 4 complete rows across both columns to run a test.")

    a_numeric = pd.api.types.is_numeric_dtype(sub[col_a])
    b_numeric = pd.api.types.is_numeric_dtype(sub[col_b])

    if a_numeric and b_numeric:
        stat, p_value = stats.pearsonr(sub[col_a], sub[col_b])
        test_name = "Pearson Correlation"
        basis = f"Pearson correlation between {col_a} and {col_b} across {len(sub)} rows."
    elif a_numeric != b_numeric:
        numeric_col, group_col = (col_a, col_b) if a_numeric else (col_b, col_a)
        groups = [g.dropna().to_numpy() for _, g in sub.groupby(group_col)[numeric_col]]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) < 2:
            raise HTTPException(
                status_code=400,
                detail=f"{group_col} needs at least 2 distinct groups with data to compare {numeric_col} across.",
            )
        if len(groups) == 2:
            stat, p_value = stats.ttest_ind(groups[0], groups[1], equal_var=False)
            test_name = "Two-Sample T-Test (Welch)"
            basis = f"Welch's t-test comparing {numeric_col} across the 2 groups of {group_col} ({len(sub)} rows)."
        else:
            stat, p_value = stats.f_oneway(*groups)
            test_name = "One-Way ANOVA"
            basis = f"One-way ANOVA comparing {numeric_col} across {len(groups)} groups of {group_col} ({len(sub)} rows)."
    else:
        contingency = pd.crosstab(sub[col_a], sub[col_b])
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            raise HTTPException(
                status_code=400,
                detail=f"Need at least 2 distinct categories in both {col_a} and {col_b} to run a chi-squared test.",
            )
        stat, p_value, dof, _ = stats.chi2_contingency(contingency)
        test_name = "Chi-Squared Test of Independence"
        basis = f"Chi-squared test of independence between {col_a} and {col_b} ({len(sub)} rows, {dof} degrees of freedom)."

    p_value = float(p_value)
    return {
        "testName": test_name,
        "testStatistic": round(float(stat), 4),
        "pValue": round(p_value, 6),
        "rejectNull": bool(p_value < 0.05),
        "sampleSize": int(len(sub)),
        "basis": basis,
    }


# --------------------------------------------------------------------------
# Agentic analysis mode: a fixed, whitelisted set of real functions that an
# LLM orchestration loop (server.ts) can call. Every one of these performs a
# genuine computation - real preprocessing, a real model fit, or a real
# regression-based statistic - and reuses the exact same training routine as
# /train, so an agent-proposed retrain is held to the same standard as any
# other model in this app. There is no path from here to arbitrary code
# execution: only these three operations are reachable.
# --------------------------------------------------------------------------

@app.post("/agent/retrain-with-transform")
def agent_retrain_with_transform(request: AgentRetrainTransformRequest):
    if not request.data:
        raise HTTPException(status_code=400, detail="No data provided.")
    if request.model_key not in VALID_MODEL_KEYS:
        raise HTTPException(status_code=400, detail=f"Unknown model_key '{request.model_key}'.")

    df = pd.DataFrame(request.data)
    if request.column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{request.column}' not found in data.")
    if not pd.api.types.is_numeric_dtype(df[request.column]):
        raise HTTPException(status_code=400, detail=f"Column '{request.column}' is not numeric; cannot apply a '{request.method}' transform.")

    df[request.column] = apply_column_transform(df[request.column], request.method)

    train_request = TrainRequest(
        data=df.to_dict(orient="records"),
        target=request.target,
        features=request.features,
        task_type=request.task_type,
        models=[request.model_key],
    )
    train_response = train_model(train_request)
    champion = train_response["champion"]

    return {
        "modelKey": champion["modelKey"],
        "modelName": champion["modelName"],
        "modelId": champion["modelId"],
        "primaryMetric": champion["primaryMetric"],
        "primaryMetricValue": champion["primaryMetricValue"],
        "cv": champion.get("cv"),
        "transformApplied": {"column": request.column, "method": request.method},
        "targetWasTransformed": request.column == request.target,
    }


@app.post("/agent/drop-feature")
def agent_drop_feature(request: AgentDropFeatureRequest):
    if not request.data:
        raise HTTPException(status_code=400, detail="No data provided.")
    if request.model_key not in VALID_MODEL_KEYS:
        raise HTTPException(status_code=400, detail=f"Unknown model_key '{request.model_key}'.")

    df = pd.DataFrame(request.data)
    base_features = request.features or [c for c in df.columns if c != request.target]
    if request.drop_feature not in base_features:
        raise HTTPException(status_code=400, detail=f"'{request.drop_feature}' is not a current feature.")
    remaining_features = [f for f in base_features if f != request.drop_feature]
    if not remaining_features:
        raise HTTPException(status_code=400, detail="Cannot drop the only remaining feature.")

    train_request = TrainRequest(
        data=request.data,
        target=request.target,
        features=remaining_features,
        task_type=request.task_type,
        models=[request.model_key],
    )
    train_response = train_model(train_request)
    champion = train_response["champion"]

    return {
        "modelKey": champion["modelKey"],
        "modelName": champion["modelName"],
        "modelId": champion["modelId"],
        "primaryMetric": champion["primaryMetric"],
        "primaryMetricValue": champion["primaryMetricValue"],
        "cv": champion.get("cv"),
        "droppedFeature": request.drop_feature,
        "remainingFeatures": remaining_features,
    }


@app.post("/agent/check-multicollinearity")
def agent_check_multicollinearity(request: AgentVifRequest):
    if not request.data:
        raise HTTPException(status_code=400, detail="No data provided.")

    df = pd.DataFrame(request.data)
    numeric_features = [f for f in request.features if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
    if len(numeric_features) < 2:
        return {"vifScores": [], "highRiskFeatures": [], "note": "Fewer than 2 numeric features - multicollinearity is not applicable."}

    vif_scores = compute_vif(df, numeric_features)
    return {
        "vifScores": vif_scores,
        "highRiskFeatures": [v["feature"] for v in vif_scores if v["highRisk"]],
    }
