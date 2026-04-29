"""
ml_advanced.py — Advanced ML Intelligence for DQ·INTEL
Covers:
  1. Smart target column detection
  2. Model-agnostic feature importance (MI + RF + Spearman)
  3. Feature selection recommendations
  4. Class imbalance detection & diagnosis
  5. Data leakage detection
  6. Dimensionality reduction (PCA + t-SNE readiness)
  7. Composite ML readiness score (replaces R²)
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from typing import Optional

warnings.filterwarnings("ignore")

# ── Optional heavy deps — graceful degradation if missing ─────────────────────
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

try:
    from scipy.stats import spearmanr, chi2_contingency, entropy, ks_2samp
    from scipy.spatial.distance import jensenshannon
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TARGET COLUMN DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

_TARGET_KEYWORDS = [
    "target", "label", "output", "class", "y", "outcome", "result",
    "churn", "fraud", "default", "survived", "price", "salary", "revenue",
    "diagnosis", "response", "prediction", "score", "rating", "grade",
    "status", "category", "type", "flag", "is_", "has_",
]

def detect_target_column(df: pd.DataFrame) -> dict:
    """
    Heuristically rank columns by likelihood of being a target.
    Returns the top candidate plus confidence and task type.
    """
    scores = {}
    n_rows = len(df)

    for col in df.columns:
        s = 0
        col_lower = col.lower().replace(" ", "_")
        series = df[col].dropna()
        n_unique = series.nunique()
        dtype = df[col].dtype

        # Keyword match
        for kw in _TARGET_KEYWORDS:
            if kw in col_lower:
                s += 30
                break

        # Last column heuristic (common in Kaggle/sklearn datasets)
        if col == df.columns[-1]:
            s += 15

        # Binary columns are strong target candidates
        if n_unique == 2:
            s += 25

        # Low cardinality categorical (classification target)
        if dtype == object and 2 <= n_unique <= 20:
            s += 20

        # Integer with low cardinality
        if dtype in (np.int64, np.int32, int) and 2 <= n_unique <= 10:
            s += 18

        # Continuous float — regression target
        if dtype in (np.float64, np.float32, float) and n_unique > 20:
            s += 10

        # Penalize ID-like columns
        if n_unique == n_rows:
            s -= 40
        if any(kw in col_lower for kw in ["id", "uuid", "key", "index", "timestamp", "date"]):
            s -= 35

        scores[col] = max(s, 0)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if not ranked or ranked[0][1] == 0:
        return {"detected": False, "col": None, "confidence": 0, "task_type": None, "ranked": []}

    best_col, best_score = ranked[0]
    confidence = min(round(best_score / 80 * 100), 99)

    series = df[best_col].dropna()
    n_unique = series.nunique()
    if n_unique == 2:
        task_type = "binary_classification"
    elif n_unique <= 20 or df[best_col].dtype == object:
        task_type = "multiclass_classification"
    else:
        task_type = "regression"

    return {
        "detected": True,
        "col": best_col,
        "confidence": confidence,
        "task_type": task_type,
        "task_label": {
            "binary_classification": "Binary Classification",
            "multiclass_classification": "Multiclass Classification",
            "regression": "Regression",
        }[task_type],
        "ranked": [{"col": c, "score": s} for c, s in ranked[:5]],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE IMPORTANCE (model-agnostic, 3-method ensemble)
# ═══════════════════════════════════════════════════════════════════════════════

def _encode_df(df: pd.DataFrame, target_col: str):
    """Return X (numeric), y arrays after label-encoding categoricals."""
    feature_cols = [c for c in df.columns if c != target_col]
    X_raw = df[feature_cols].copy()

    # Encode categoricals
    for col in X_raw.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X_raw[col] = le.fit_transform(X_raw[col].astype(str))

    # Impute
    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(X_raw)

    # Encode target
    y_raw = df[target_col].copy()
    if y_raw.dtype == object:
        le = LabelEncoder()
        y = le.fit_transform(y_raw.astype(str))
    else:
        y = y_raw.fillna(y_raw.median()).values

    return X, y, feature_cols


def compute_feature_importance(df: pd.DataFrame, target_col: str, task_type: str) -> list[dict]:
    """
    Ensemble of 3 methods:
      - Mutual Information (captures non-linear dependencies)
      - Random Forest importance (gini/variance reduction)
      - Spearman rank correlation (monotonic, robust)
    Returns a ranked list of features with per-method scores and ensemble rank.
    """
    if not SKLEARN_OK or not SCIPY_OK:
        return []

    sample = df.dropna(subset=[target_col])
    if len(sample) < 30:
        return []

    # Sub-sample for speed on large datasets
    if len(sample) > 5000:
        sample = sample.sample(5000, random_state=42)

    X, y, feature_cols = _encode_df(sample, target_col)
    is_clf = "classification" in task_type
    n_features = X.shape[1]

    results = {col: {"col": col, "mi": 0.0, "rf": 0.0, "spearman": 0.0} for col in feature_cols}

    # — Mutual Information —
    try:
        mi_fn = mutual_info_classif if is_clf else mutual_info_regression
        mi_scores = mi_fn(X, y, random_state=42)
        mi_max = mi_scores.max() or 1.0
        for i, col in enumerate(feature_cols):
            results[col]["mi"] = round(float(mi_scores[i] / mi_max), 4)
    except Exception:
        pass

    # — Random Forest importance —
    try:
        rf = (RandomForestClassifier if is_clf else RandomForestRegressor)(
            n_estimators=100, max_depth=6, random_state=42, n_jobs=-1
        )
        rf.fit(X, y)
        rf_scores = rf.feature_importances_
        rf_max = rf_scores.max() or 1.0
        for i, col in enumerate(feature_cols):
            results[col]["rf"] = round(float(rf_scores[i] / rf_max), 4)
    except Exception:
        pass

    # — Spearman correlation —
    try:
        for i, col in enumerate(feature_cols):
            rho, _ = spearmanr(X[:, i], y)
            results[col]["spearman"] = round(abs(float(rho)) if not np.isnan(rho) else 0.0, 4)
    except Exception:
        pass

    # — Ensemble score (weighted average) —
    for col in feature_cols:
        r = results[col]
        r["ensemble"] = round(0.40 * r["mi"] + 0.40 * r["rf"] + 0.20 * r["spearman"], 4)
        r["ensemble_pct"] = round(r["ensemble"] * 100, 1)

    ranked = sorted(results.values(), key=lambda x: x["ensemble"], reverse=True)

    # Add rank and recommendation
    for i, r in enumerate(ranked):
        r["rank"] = i + 1
        if r["ensemble"] >= 0.6:
            r["recommendation"] = "Keep — strong signal"
            r["rec_level"] = "high"
        elif r["ensemble"] >= 0.25:
            r["recommendation"] = "Keep — moderate signal"
            r["rec_level"] = "medium"
        elif r["ensemble"] >= 0.08:
            r["recommendation"] = "Consider — weak signal"
            r["rec_level"] = "low"
        else:
            r["recommendation"] = "Drop — near-zero importance"
            r["rec_level"] = "drop"

    return ranked


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CLASS IMBALANCE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_class_imbalance(df: pd.DataFrame, target_col: str, task_type: str) -> dict:
    """
    Full imbalance analysis:
      - Imbalance ratio
      - Effective number of samples per class
      - Shannon entropy of class distribution
      - Severity rating & recommended strategy
    """
    if "regression" in task_type:
        # For regression: check target distribution skew
        series = df[target_col].dropna()
        skew = float(series.skew())
        kurt = float(series.kurtosis())
        severity = "none"
        if abs(skew) > 2:
            severity = "high"
        elif abs(skew) > 1:
            severity = "medium"
        elif abs(skew) > 0.5:
            severity = "low"

        return {
            "task_type": "regression",
            "skew": round(skew, 3),
            "kurtosis": round(kurt, 3),
            "severity": severity,
            "recommendation": _regression_skew_rec(skew),
            "classes": [],
        }

    counts = df[target_col].value_counts()
    total = counts.sum()
    n_classes = len(counts)

    class_info = []
    for cls, cnt in counts.items():
        class_info.append({
            "class": str(cls),
            "count": int(cnt),
            "pct": round(cnt / total * 100, 2),
        })

    # Imbalance ratio (majority / minority)
    ratio = float(counts.iloc[0] / counts.iloc[-1]) if counts.iloc[-1] > 0 else float("inf")

    # Shannon entropy (max entropy = log2(n_classes) = perfectly balanced)
    probs = counts.values / total
    shannon = float(entropy(probs, base=2)) if SCIPY_OK else 0.0
    max_entropy = np.log2(n_classes) if n_classes > 1 else 1.0
    balance_score = round(shannon / max_entropy * 100, 1)  # 100 = perfectly balanced

    # Severity
    if ratio >= 20 or balance_score < 40:
        severity = "critical"
    elif ratio >= 10 or balance_score < 60:
        severity = "high"
    elif ratio >= 4 or balance_score < 75:
        severity = "medium"
    else:
        severity = "none"

    strategies = _imbalance_strategies(ratio, n_classes, severity)

    return {
        "task_type": task_type,
        "n_classes": n_classes,
        "imbalance_ratio": round(ratio, 2),
        "balance_score": balance_score,
        "severity": severity,
        "shannon_entropy": round(shannon, 3),
        "max_entropy": round(max_entropy, 3),
        "classes": class_info,
        "strategies": strategies,
    }


def _regression_skew_rec(skew: float) -> str:
    if abs(skew) <= 0.5:
        return "Target is approximately normal — no transformation needed."
    if abs(skew) <= 1.0:
        return "Mild skew — consider sqrt or Box-Cox transformation for linear models."
    if abs(skew) <= 2.0:
        return "Moderate skew — apply log1p (if positive) or Yeo-Johnson transform."
    return "Heavy skew — log1p or Yeo-Johnson strongly recommended. Check for outliers pulling the tail."


def _imbalance_strategies(ratio: float, n_classes: int, severity: str) -> list[str]:
    if severity == "none":
        return ["Dataset is well-balanced — no resampling needed."]
    strategies = []
    if ratio < 10:
        strategies.append("class_weight='balanced' in sklearn estimators — zero overhead, often sufficient.")
    strategies.append("SMOTE (imbalanced-learn) — synthetic oversampling of minority class(es).")
    if n_classes == 2:
        strategies.append("Adjust classification threshold (e.g. 0.3 instead of 0.5) to improve minority recall.")
        strategies.append("Use PR-AUC or F1 as your metric instead of accuracy.")
    else:
        strategies.append("Use macro-averaged F1 or balanced accuracy as evaluation metric.")
    if ratio >= 20:
        strategies.append("Consider undersampling majority class (RandomUnderSampler) to speed up training.")
    return strategies


# ═══════════════════════════════════════════════════════════════════════════════
# 4. DATA LEAKAGE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_data_leakage(df: pd.DataFrame, target_col: str, task_type: str) -> list[dict]:
    """
    Detects potential data leakage signals:
      - Near-perfect correlation with target (suspiciously high MI or Spearman)
      - Columns derived from target (name-based heuristics)
      - Future-information columns (date/time after the event)
      - Constant-per-target-class columns
    """
    if not SCIPY_OK:
        return []

    leakage_flags = []
    sample = df.dropna(subset=[target_col])
    if len(sample) > 5000:
        sample = sample.sample(5000, random_state=42)

    is_clf = "classification" in task_type

    # Encode target
    y_raw = sample[target_col]
    if y_raw.dtype == object:
        le = LabelEncoder()
        y = le.fit_transform(y_raw.astype(str)) if SKLEARN_OK else y_raw.values
    else:
        y = y_raw.fillna(y_raw.median()).values

    for col in df.columns:
        if col == target_col:
            continue

        col_lower = col.lower()
        flags = []

        # 1. Name-based: sounds derived from target
        derived_kws = ["_pred", "_hat", "_score", "_proba", "_probability",
                       "_result", "_output", "pred_", "predicted_"]
        if any(kw in col_lower for kw in derived_kws):
            flags.append("Name suggests it may be derived from the target (prediction artifact)")

        # 2. Statistical leakage: suspiciously high correlation / MI
        try:
            series = sample[col].copy()
            if series.dtype == object:
                le2 = LabelEncoder()
                x = le2.fit_transform(series.fillna("__NA__").astype(str))
            else:
                x = series.fillna(series.median()).values

            if SKLEARN_OK and is_clf:
                mi = mutual_info_classif(x.reshape(-1, 1), y, random_state=42)[0]
                if mi > 1.5:  # extremely high mutual information
                    flags.append(f"Mutual information = {mi:.2f} nats — suspiciously high, possible leakage")
            elif SKLEARN_OK:
                mi = mutual_info_regression(x.reshape(-1, 1), y, random_state=42)[0]
                if mi > 1.5:
                    flags.append(f"Mutual information = {mi:.2f} nats — suspiciously high, possible leakage")

            rho, pval = spearmanr(x, y)
            if abs(rho) > 0.97 and pval < 0.01:
                flags.append(f"Spearman |r| = {abs(rho):.3f} — near-perfect correlation with target")
        except Exception:
            pass

        # 3. Constant within target class (perfect predictor)
        try:
            if is_clf:
                group_nunique = sample.groupby(target_col)[col].nunique()
                if (group_nunique <= 1).all() and sample[col].nunique() > 1:
                    flags.append("Column is constant within each target class — likely a direct label encoding")
        except Exception:
            pass

        # 4. Future-date heuristic
        future_kws = ["future", "after", "post_", "end_date", "closing", "final"]
        if any(kw in col_lower for kw in future_kws):
            flags.append("Column name suggests future information that wouldn't be available at prediction time")

        if flags:
            leakage_flags.append({
                "col": col,
                "flags": flags,
                "severity": "high" if len(flags) > 1 else "medium",
            })

    return sorted(leakage_flags, key=lambda x: len(x["flags"]), reverse=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. PCA / DIMENSIONALITY REDUCTION READINESS
# ═══════════════════════════════════════════════════════════════════════════════

def pca_analysis(df: pd.DataFrame) -> dict:
    """
    Runs PCA on numeric columns and returns:
      - Explained variance per component
      - Components needed to reach 80%, 90%, 95% variance
      - Redundancy score (how much dimensionality can be reduced)
      - Top feature loadings per component
      - t-SNE readiness verdict
    """
    if not SKLEARN_OK:
        return {"available": False, "reason": "scikit-learn not installed"}

    num_df = df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    num_df = num_df.fillna(num_df.median())

    n_features = num_df.shape[1]
    n_samples = num_df.shape[0]

    if n_features < 3:
        return {"available": False, "reason": "Need at least 3 numeric columns for PCA"}
    if n_samples < 10:
        return {"available": False, "reason": "Too few rows for PCA"}

    # Sub-sample for speed
    if n_samples > 10000:
        num_df = num_df.sample(10000, random_state=42)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(num_df)

    n_components = min(n_features, n_samples - 1, 20)
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X_scaled)

    exp_var = pca.explained_variance_ratio_
    cum_var = np.cumsum(exp_var)

    def components_for_threshold(t):
        idx = np.searchsorted(cum_var, t)
        return int(min(idx + 1, n_components))

    c80 = components_for_threshold(0.80)
    c90 = components_for_threshold(0.90)
    c95 = components_for_threshold(0.95)

    # Redundancy score: how much we can reduce dimensions to reach 95% variance
    redundancy = round((1 - c95 / n_features) * 100, 1)

    # Top loadings per component (top 3 features)
    feature_names = list(num_df.columns)
    component_loadings = []
    for i in range(min(5, n_components)):
        loading = pca.components_[i]
        top_idx = np.argsort(np.abs(loading))[::-1][:3]
        component_loadings.append({
            "component": i + 1,
            "explained_variance_pct": round(float(exp_var[i]) * 100, 2),
            "cumulative_variance_pct": round(float(cum_var[i]) * 100, 2),
            "top_features": [
                {"feature": feature_names[j], "loading": round(float(loading[j]), 3)}
                for j in top_idx
            ],
        })

    # t-SNE readiness
    tsne_ready = n_samples >= 100 and n_features >= 4
    tsne_recommendation = (
        "Ready — use t-SNE on the first 50 PCA components to speed up computation."
        if n_features > 50 and tsne_ready
        else "Ready — t-SNE can run directly on your numeric features." if tsne_ready
        else "Not recommended — t-SNE needs at least 100 rows and 4 features."
    )

    return {
        "available": True,
        "n_features": n_features,
        "n_components_computed": n_components,
        "components_for_80pct": c80,
        "components_for_90pct": c90,
        "components_for_95pct": c95,
        "redundancy_score": redundancy,
        "explained_variance": [round(float(v) * 100, 2) for v in exp_var],
        "cumulative_variance": [round(float(v) * 100, 2) for v in cum_var],
        "component_loadings": component_loadings,
        "tsne_ready": tsne_ready,
        "tsne_recommendation": tsne_recommendation,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MODEL-AGNOSTIC ML READINESS SCORE (replaces R²)
# ═══════════════════════════════════════════════════════════════════════════════

def ml_readiness_advanced(df: pd.DataFrame, analysis: dict, target_col: Optional[str] = None) -> dict:
    """
    Comprehensive ML readiness score built from 7 sub-scores.
    Each sub-score is 0–100. Final score is weighted average.
    No R² anywhere.
    """
    cols = analysis["columns"]
    n_rows = analysis["total_rows"]
    n_cols = analysis["num_cols"]

    sub_scores = {}
    factors = []

    # ── 1. Completeness (30%) ─────────────────────────────────────────────────
    avg_missing = np.mean([c["missing_pct"] for c in cols])
    completeness = max(0, 100 - avg_missing * 2)
    sub_scores["completeness"] = round(completeness, 1)
    if avg_missing > 20:
        factors.append({"label": "High average missingness", "detail": f"{avg_missing:.1f}% average missing — imputation required before training.", "severity": "high"})
    elif avg_missing > 5:
        factors.append({"label": "Moderate missingness", "detail": f"{avg_missing:.1f}% average missing — use median/mode imputation or IterativeImputer.", "severity": "medium"})

    # ── 2. Sample size adequacy (20%) ─────────────────────────────────────────
    n_numeric = sum(1 for c in cols if c["type"] == "numeric")
    n_cat = sum(1 for c in cols if c["type"] == "categorical")
    effective_features = n_numeric + n_cat * 2  # categoricals expand with encoding

    if n_rows >= effective_features * 100:
        sample_score = 100
    elif n_rows >= effective_features * 30:
        sample_score = 80
    elif n_rows >= effective_features * 10:
        sample_score = 55
    elif n_rows >= 100:
        sample_score = 30
    else:
        sample_score = 10

    sub_scores["sample_size"] = sample_score
    if sample_score < 55:
        factors.append({
            "label": "Insufficient rows for feature count",
            "detail": f"{n_rows:,} rows for ~{effective_features} effective features. Aim for ≥{effective_features*30:,} rows, or reduce feature count.",
            "severity": "high" if sample_score < 30 else "medium",
        })

    # ── 3. Feature quality via Mutual Information (20%) ──────────────────────
    mi_score = 70  # default when no target
    if target_col and SKLEARN_OK and SCIPY_OK:
        try:
            importance = compute_feature_importance(df, target_col, "binary_classification")
            if importance:
                ensemble_scores = [f["ensemble"] for f in importance]
                pct_useful = sum(1 for s in ensemble_scores if s >= 0.08) / len(ensemble_scores)
                mi_score = round(pct_useful * 100, 1)
                if pct_useful < 0.4:
                    factors.append({
                        "label": "Low feature signal",
                        "detail": f"Only {pct_useful*100:.0f}% of features have meaningful signal with target. Consider feature engineering.",
                        "severity": "high" if pct_useful < 0.2 else "medium",
                    })
        except Exception:
            pass
    sub_scores["feature_quality"] = mi_score

    # ── 4. Cardinality & encoding readiness (10%) ─────────────────────────────
    high_card_cols = [c for c in cols if c["type"] == "categorical" and c["stats"].get("n_unique", 0) > 50]
    card_penalty = len(high_card_cols) / max(n_cols, 1) * 100
    card_score = max(0, 100 - card_penalty * 3)
    sub_scores["cardinality"] = round(card_score, 1)
    if high_card_cols:
        factors.append({
            "label": "High-cardinality categoricals",
            "detail": f"{len(high_card_cols)} column(s) with >50 unique values: {', '.join(c['col'] for c in high_card_cols[:3])}. Use target encoding or embeddings.",
            "severity": "medium",
        })

    # ── 5. Outlier burden (10%) ───────────────────────────────────────────────
    outlier_pct_per_col = [
        c["outliers"] / max(analysis["sample_size"], 1) * 100
        for c in cols if c["type"] == "numeric"
    ]
    avg_outlier_pct = np.mean(outlier_pct_per_col) if outlier_pct_per_col else 0
    outlier_score = max(0, 100 - avg_outlier_pct * 5)
    sub_scores["outliers"] = round(outlier_score, 1)
    if avg_outlier_pct > 10:
        factors.append({
            "label": "Heavy outlier presence",
            "detail": f"Average {avg_outlier_pct:.1f}% outliers across numeric columns. Consider RobustScaler or Winsorization.",
            "severity": "medium" if avg_outlier_pct < 20 else "high",
        })

    # ── 6. Duplicate burden (5%) ──────────────────────────────────────────────
    dup_pct = analysis["dupes_estimated"] / max(n_rows, 1) * 100
    dup_score = max(0, 100 - dup_pct * 10)
    sub_scores["duplicates"] = round(dup_score, 1)
    if dup_pct > 5:
        factors.append({
            "label": "Duplicate rows",
            "detail": f"~{dup_pct:.1f}% duplicate rows. Remove before splitting train/test to avoid data leakage.",
            "severity": "medium",
        })

    # ── 7. Type consistency (5%) ──────────────────────────────────────────────
    incon_count = sum(1 for c in cols if c.get("inconsistent", 0) > 0)
    type_score = max(0, 100 - (incon_count / max(n_cols, 1)) * 200)
    sub_scores["type_consistency"] = round(type_score, 1)
    if incon_count > 0:
        factors.append({
            "label": "Type inconsistencies",
            "detail": f"{incon_count} column(s) have mixed types or format issues — fix before encoding.",
            "severity": "low",
        })

    # ── Weighted composite ────────────────────────────────────────────────────
    weights = {
        "completeness":     0.30,
        "sample_size":      0.20,
        "feature_quality":  0.20,
        "cardinality":      0.10,
        "outliers":         0.10,
        "duplicates":       0.05,
        "type_consistency": 0.05,
    }
    final_score = sum(sub_scores[k] * w for k, w in weights.items())
    final_score = round(final_score, 1)

    label = "Ready" if final_score >= 80 else "Needs Work" if final_score >= 60 else "Not Ready"

    high_missing_cols = [c["col"] for c in cols if c["missing_pct"] > 20]

    return {
        "score": final_score,
        "label": label,
        "sub_scores": sub_scores,
        "weights": weights,
        "factors": sorted(factors, key=lambda f: {"high": 3, "medium": 2, "low": 1}[f["severity"]], reverse=True),
        "num_features": n_numeric,
        "cat_features": n_cat,
        "high_missing_cols": high_missing_cols,
        "target_used": target_col is not None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 7. MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def run_advanced_ml(df: pd.DataFrame, analysis: dict, target_col: Optional[str] = None) -> dict:
    """
    Master function — run all 6 ML analyses and return a single result dict.
    Call this from app.py instead of the old ml_readiness().
    """
    # Auto-detect target if not provided
    target_detection = detect_target_column(df)
    resolved_target = target_col or (target_detection["col"] if target_detection["detected"] else None)

    # Run all analyses
    readiness     = ml_readiness_advanced(df, analysis, resolved_target)
    importance    = compute_feature_importance(df, resolved_target, target_detection.get("task_type", "binary_classification")) if resolved_target and SKLEARN_OK else []
    imbalance     = detect_class_imbalance(df, resolved_target, target_detection.get("task_type", "binary_classification")) if resolved_target else None
    leakage       = detect_data_leakage(df, resolved_target, target_detection.get("task_type", "binary_classification")) if resolved_target and SCIPY_OK else []
    pca           = pca_analysis(df)

    return {
        "target_detection": target_detection,
        "resolved_target":  resolved_target,
        "readiness":        readiness,
        "importance":       importance,
        "imbalance":        imbalance,
        "leakage":          leakage,
        "pca":              pca,
        "sklearn_available": SKLEARN_OK,
        "scipy_available":  SCIPY_OK,
    }