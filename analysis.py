"""
DQ·INTEL — Data Quality Analysis Engine
Pure Python/pandas implementation replacing the JS heuristics.
"""

import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
import re
from typing import Any


# ── TYPE INFERENCE ──────────────────────────────────────────────────────────

def infer_column_type(series: pd.Series) -> str:
    """
    Infer semantic type of a column.
    Returns: 'numeric', 'date', 'categorical', 'boolean', 'id', 'text'
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return "unknown"

    # Boolean check
    unique_lower = set(str(v).strip().lower() for v in non_null.unique())
    if unique_lower <= {"true", "false", "yes", "no", "1", "0", "t", "f", "y", "n"}:
        return "boolean"

    # Numeric check
    try:
        numeric = pd.to_numeric(non_null, errors="coerce")
        numeric_ratio = numeric.notna().sum() / len(non_null)
        if numeric_ratio >= 0.9:
            # ID check: numeric, high cardinality, integer-like
            if numeric_ratio >= 0.99:
                vals = numeric.dropna()
                if (vals == vals.astype(int)).all():
                    cardinality = len(vals.unique()) / len(vals)
                    if cardinality > 0.95:
                        return "id"
            return "numeric"
    except Exception:
        pass

    # Date check
    date_patterns = [
        r"\d{4}-\d{2}-\d{2}",
        r"\d{2}/\d{2}/\d{4}",
        r"\d{2}-\d{2}-\d{4}",
        r"\d{1,2}/\d{1,2}/\d{2,4}",
    ]
    sample = non_null.astype(str).head(50)
    date_matches = sum(
        1 for v in sample
        if any(re.search(p, v) for p in date_patterns)
    )
    if date_matches / max(len(sample), 1) > 0.7:
        try:
            pd.to_datetime(non_null.head(100), infer_datetime_format=True, errors="raise")
            return "date"
        except Exception:
            pass

    # High cardinality text vs categorical
    cardinality = len(non_null.unique()) / len(non_null)
    avg_len = non_null.astype(str).str.len().mean()
    if cardinality > 0.7 and avg_len > 20:
        return "text"

    return "categorical"


# ── MISSING VALUE DETECTION ──────────────────────────────────────────────────

MISSING_PATTERNS = re.compile(
    r"^\s*$|^(nan|none|null|na|n/a|#n/a|missing|unknown|-|--|---|\?)$",
    re.IGNORECASE,
)

def is_missing(val: Any) -> bool:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return True
    return bool(MISSING_PATTERNS.match(str(val).strip()))


# ── CONSISTENCY CHECKS ───────────────────────────────────────────────────────

DATE_FORMATS = [
    r"^\d{4}-\d{2}-\d{2}$",          # ISO
    r"^\d{2}/\d{2}/\d{4}$",          # US
    r"^\d{2}-\d{2}-\d{4}$",          # EU dash
    r"^\d{1,2}/\d{1,2}/\d{2}$",      # short US
    r"^\d{4}/\d{2}/\d{2}$",          # ISO slash
]

def check_consistency(series: pd.Series, col_type: str) -> tuple[int, str]:
    """
    Returns (inconsistent_count, description).
    Checks mixed date formats, mixed case in categoricals, mixed numeric/text.
    """
    non_null = series.dropna().astype(str)
    if len(non_null) == 0:
        return 0, ""

    if col_type == "date":
        # Detect mixed date formats
        format_hits = {pat: 0 for pat in DATE_FORMATS}
        for v in non_null:
            for pat in DATE_FORMATS:
                if re.match(pat, v.strip()):
                    format_hits[pat] += 1
                    break
        formats_used = sum(1 for c in format_hits.values() if c > 0)
        if formats_used > 1:
            minority = len(non_null) - max(format_hits.values())
            return minority, "mixed date formats"
        return 0, ""

    if col_type == "categorical":
        # Mixed case: same value with different casing
        lower_counts = non_null.str.strip().str.lower().value_counts()
        raw_counts = non_null.str.strip().value_counts()
        if len(lower_counts) < len(raw_counts):
            inconsistent = len(non_null) - raw_counts.max() * len(raw_counts)
            # count values that appear in multiple case variants
            case_issues = 0
            for lower_val, total_count in lower_counts.items():
                variants = [v for v in raw_counts.index if v.lower() == lower_val]
                if len(variants) > 1:
                    case_issues += total_count - raw_counts[variants].max()
            if case_issues > 0:
                return case_issues, "mixed case variants"

        # Leading/trailing whitespace
        ws_issues = non_null[non_null != non_null.str.strip()].shape[0]
        if ws_issues > 0:
            return ws_issues, "whitespace inconsistencies"

    if col_type == "numeric":
        # Values that look like numbers but have formatting (e.g. "1,000", "$50")
        numeric = pd.to_numeric(non_null, errors="coerce")
        non_numeric = numeric.isna().sum()
        # strip common formatting and re-check
        cleaned = non_null.str.replace(r"[\$,€£%\s]", "", regex=True)
        cleaned_numeric = pd.to_numeric(cleaned, errors="coerce")
        format_issues = cleaned_numeric.notna().sum() - numeric.notna().sum()
        if format_issues > 0:
            return int(format_issues), "numeric formatting (commas/symbols)"

    return 0, ""


# ── OUTLIER DETECTION ────────────────────────────────────────────────────────

def detect_outliers(series: pd.Series) -> tuple[int, float, float]:
    """
    IQR 1.5x method. Returns (count, lower_fence, upper_fence).
    """
    clean = pd.to_numeric(series.dropna(), errors="coerce").dropna()
    if len(clean) < 4:
        return 0, float("nan"), float("nan")
    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = ((clean < lower) | (clean > upper)).sum()
    return int(outliers), float(lower), float(upper)


# ── COLUMN STATS ─────────────────────────────────────────────────────────────

def column_stats(series: pd.Series, col_type: str) -> dict:
    """Compute descriptive stats for a column."""
    non_null = series.dropna()
    result: dict = {}

    if col_type == "numeric":
        nums = pd.to_numeric(non_null, errors="coerce").dropna()
        if len(nums) == 0:
            return result
        result = {
            "mean": float(nums.mean()),
            "median": float(nums.median()),
            "std": float(nums.std()),
            "min": float(nums.min()),
            "max": float(nums.max()),
            "q1": float(nums.quantile(0.25)),
            "q3": float(nums.quantile(0.75)),
            "skewness": float(scipy_stats.skew(nums)),
            "kurtosis": float(scipy_stats.kurtosis(nums)),
        }
        # Histogram bins
        counts, edges = np.histogram(nums, bins=min(15, len(nums.unique())))
        result["histogram"] = {
            "counts": counts.tolist(),
            "edges": edges.tolist(),
        }

    elif col_type in ("categorical", "boolean", "id"):
        vc = non_null.astype(str).str.strip().value_counts()
        result = {
            "top_values": vc.head(10).to_dict(),
            "cardinality": int(non_null.nunique()),
        }

    elif col_type == "date":
        try:
            parsed = pd.to_datetime(non_null, infer_datetime_format=True, errors="coerce").dropna()
            if len(parsed):
                result = {
                    "min_date": str(parsed.min().date()),
                    "max_date": str(parsed.max().date()),
                    "span_days": (parsed.max() - parsed.min()).days,
                }
        except Exception:
            pass

    return result


# ── CORE ANALYSIS ────────────────────────────────────────────────────────────

def analyze_dataset(df: pd.DataFrame, file_name: str = "") -> dict:
    """
    Full DQ analysis. Returns a dict with all metrics.
    Samples up to 100k rows for performance but scores on full row count.
    """
    total_rows = len(df)
    sample_size = min(total_rows, 100_000)
    sample = df.sample(n=sample_size, random_state=42) if total_rows > sample_size else df

    columns = []
    tot_missing = 0
    tot_inconsistent = 0
    tot_invalid = 0

    for col in sample.columns:
        series = sample[col]
        col_type = infer_column_type(series)

        # Missing
        missing_mask = series.apply(is_missing)
        missing_count = int(missing_mask.sum())
        missing_pct = (missing_count / sample_size) * 100

        non_null = series[~missing_mask]

        # Consistency
        incon_count, incon_desc = check_consistency(non_null, col_type)

        # Outliers (numeric only)
        outlier_count, lower_fence, upper_fence = (0, float("nan"), float("nan"))
        if col_type == "numeric":
            outlier_count, lower_fence, upper_fence = detect_outliers(non_null)

        # Stats
        stats = column_stats(non_null, col_type)
        if col_type == "numeric" and not np.isnan(lower_fence):
            stats["lower_fence"] = lower_fence
            stats["upper_fence"] = upper_fence

        tot_missing += missing_count
        tot_inconsistent += incon_count
        tot_invalid += outlier_count

        columns.append({
            "col": col,
            "type": col_type,
            "missing": missing_count,
            "missing_pct": round(missing_pct, 2),
            "non_missing": int(non_null.shape[0]),
            "inconsistent": incon_count,
            "inconsistent_desc": incon_desc,
            "outliers": outlier_count,
            "stats": stats,
        })

    # Duplicate detection
    dupe_count = int(sample.duplicated().sum())
    dupe_est = round(dupe_count * (total_rows / sample_size))

    # DQS dimensions
    total_cells = len(sample.columns) * sample_size
    C  = 100 * (1 - tot_missing / max(total_cells, 1))         # Completeness
    Co = 100 * (1 - tot_inconsistent / max(total_cells, 1))    # Consistency
    V  = 100 * (1 - tot_invalid / max(total_cells, 1))         # Validity
    U  = 100 * (1 - dupe_count / max(sample_size, 1))          # Uniqueness
    DQS = 0.30 * C + 0.25 * Co + 0.25 * V + 0.20 * U

    # Hotspots
    def impact_score(c):
        return c["missing_pct"] * 0.3 + (c["outliers"] / max(sample_size, 1)) * 25 + (c["inconsistent"] / max(sample_size, 1)) * 25

    hotspots = sorted(columns, key=impact_score, reverse=True)[:5]

    # Correlation matrix (numeric cols only)
    num_cols = [c["col"] for c in columns if c["type"] == "numeric"]
    corr_matrix = None
    if len(num_cols) >= 2:
        numeric_df = sample[num_cols].apply(pd.to_numeric, errors="coerce")
        corr_matrix = numeric_df.corr().round(3).to_dict()

    return {
        "file_name": file_name,
        "total_rows": total_rows,
        "sample_size": sample_size,
        "num_cols": len(df.columns),
        "columns": columns,
        "dupes_in_sample": dupe_count,
        "dupes_estimated": dupe_est,
        "tot_missing": tot_missing,
        "total_cells": total_cells,
        "C": round(C, 2),
        "Co": round(Co, 2),
        "V": round(V, 2),
        "U": round(U, 2),
        "DQS": round(DQS, 2),
        "hotspots": [h["col"] for h in hotspots[:3]],
        "corr_matrix": corr_matrix,
        "num_col_names": num_cols,
    }


# ── RECOMMENDATIONS ──────────────────────────────────────────────────────────

def generate_recommendations(analysis: dict) -> list[dict]:
    recs = []
    total_rows = analysis["total_rows"]

    for c in analysis["columns"]:
        col = c["col"]
        col_type = c["type"]

        if c["missing_pct"] > 30:
            strategy = "Median imputation" if col_type == "numeric" else "Mode imputation or drop column"
            code = (
                f"df['{col}'].fillna(df['{col}'].median(), inplace=True)"
                if col_type == "numeric"
                else f"df['{col}'].fillna(df['{col}'].mode()[0], inplace=True)"
            )
            recs.append({
                "col": col, "impact": "High",
                "issue": f"High missing values ({c['missing_pct']:.1f}%)",
                "strategy": strategy,
                "reason": "Median is robust to outliers. Mode preserves categorical distribution.",
                "code": code,
            })
        elif c["missing_pct"] > 5:
            recs.append({
                "col": col, "impact": "Medium",
                "issue": f"Moderate missing values ({c['missing_pct']:.1f}%)",
                "strategy": "Mean or median imputation" if col_type == "numeric" else "Mode imputation",
                "reason": "Low enough to impute without distorting overall distribution.",
                "code": f"df['{col}'].fillna(df['{col}'].median(), inplace=True)",
            })

        if c["outliers"] > 0 and col_type == "numeric":
            recs.append({
                "col": col, "impact": "High" if c["outliers"] > 5 else "Medium",
                "issue": f"{c['outliers']} outlier(s) detected (IQR 1.5× rule)",
                "strategy": "Winsorize at IQR fences or remove rows",
                "reason": "Extreme values destabilize model training and skew summaries.",
                "code": (
                    f"Q1 = df['{col}'].quantile(0.25)\n"
                    f"Q3 = df['{col}'].quantile(0.75)\n"
                    f"IQR = Q3 - Q1\n"
                    f"df['{col}'] = df['{col}'].clip(lower=Q1-1.5*IQR, upper=Q3+1.5*IQR)"
                ),
            })

        if c["inconsistent"] > 0:
            desc = c.get("inconsistent_desc", "formatting inconsistencies")
            if "case" in desc:
                code = f"df['{col}'] = df['{col}'].str.strip().str.lower()"
            elif "date" in desc:
                code = f"df['{col}'] = pd.to_datetime(df['{col}'], infer_datetime_format=True)"
            elif "numeric" in desc:
                code = f"df['{col}'] = pd.to_numeric(df['{col}'].str.replace(r'[\\$,€£%]', '', regex=True), errors='coerce')"
            else:
                code = f"df['{col}'] = df['{col}'].str.strip()"
            recs.append({
                "col": col, "impact": "Low",
                "issue": f"{c['inconsistent']} {desc}",
                "strategy": "Normalize and strip",
                "reason": "Inconsistent formatting creates artificial category splits.",
                "code": code,
            })

    if analysis["dupes_estimated"] > 0:
        recs.append({
            "col": "All columns", "impact": "High" if analysis["dupes_estimated"] > total_rows * 0.05 else "Medium",
            "issue": f"~{analysis['dupes_estimated']:,} duplicate rows found",
            "strategy": "Drop duplicates, keep first occurrence",
            "reason": "Duplicates inflate sample size and bias evaluation metrics.",
            "code": "df.drop_duplicates(keep='first', inplace=True)",
        })

    return sorted(recs, key=lambda r: {"High": 3, "Medium": 2, "Low": 1}[r["impact"]], reverse=True)


# ── ML READINESS ─────────────────────────────────────────────────────────────

def ml_readiness(analysis: dict) -> dict:
    """Assess ML readiness across multiple dimensions."""
    num_cols = [c for c in analysis["columns"] if c["type"] == "numeric"]
    cat_cols = [c for c in analysis["columns"] if c["type"] == "categorical"]
    high_miss = [c for c in analysis["columns"] if c["missing_pct"] > 20]
    high_outlier = [c for c in analysis["columns"] if c["outliers"] > 0]

    # Readiness score (0-100)
    score = analysis["DQS"]
    # Penalize harder for ML-specific issues
    if len(high_miss) > 0:
        score -= len(high_miss) * 3
    if analysis["dupes_estimated"] / max(analysis["total_rows"], 1) > 0.05:
        score -= 10
    score = max(0, min(100, score))

    factors = []
    if analysis["C"] < 80:
        factors.append({
            "label": "Missing values",
            "detail": f"{len(high_miss)} column(s) have >20% missing — will cause errors in most sklearn estimators without imputation.",
            "severity": "high"
        })
    if high_outlier:
        factors.append({
            "label": "Outliers present",
            "detail": f"{len(high_outlier)} numeric column(s) have outliers. Tree models are robust; linear/distance models are not.",
            "severity": "medium"
        })
    if analysis["dupes_estimated"] > 0:
        pct = analysis["dupes_estimated"] / max(analysis["total_rows"], 1) * 100
        factors.append({
            "label": "Duplicate rows",
            "detail": f"~{pct:.1f}% duplicates — will leak into validation splits if not removed before train/test split.",
            "severity": "high" if pct > 5 else "low"
        })
    if len(cat_cols) > 0:
        factors.append({
            "label": "Categorical encoding needed",
            "detail": f"{len(cat_cols)} categorical column(s) require encoding (one-hot or label encoding) before most models.",
            "severity": "low"
        })
    if analysis["total_rows"] < 1000:
        factors.append({
            "label": "Small dataset",
            "detail": f"Only {analysis['total_rows']:,} rows — high risk of overfitting. Consider cross-validation.",
            "severity": "medium"
        })

    return {
        "score": round(score, 1),
        "label": "Ready" if score >= 80 else "Needs Work" if score >= 60 else "Not Ready",
        "factors": factors,
        "num_features": len(num_cols),
        "cat_features": len(cat_cols),
        "high_missing_cols": [c["col"] for c in high_miss],
    }


# ── SCHEMA DRIFT ─────────────────────────────────────────────────────────────

def compare_schemas(analysis_a: dict, analysis_b: dict, name_a: str, name_b: str) -> dict:
    cols_a = {c["col"]: c for c in analysis_a["columns"]}
    cols_b = {c["col"]: c for c in analysis_b["columns"]}

    added = [c for c in cols_b if c not in cols_a]
    removed = [c for c in cols_a if c not in cols_b]
    common = [c for c in cols_a if c in cols_b]

    changes = []
    for col in common:
        ca, cb = cols_a[col], cols_b[col]
        diffs = []
        if ca["type"] != cb["type"]:
            diffs.append(f"type: {ca['type']} → {cb['type']}")
        miss_diff = abs(ca["missing_pct"] - cb["missing_pct"])
        if miss_diff > 5:
            diffs.append(f"missing: {ca['missing_pct']:.1f}% → {cb['missing_pct']:.1f}%")
        # Distribution shift (numeric)
        if ca["type"] == "numeric" and ca["stats"] and cb["stats"]:
            mean_a, mean_b = ca["stats"].get("mean", 0), cb["stats"].get("mean", 0)
            std_a = ca["stats"].get("std", 1) or 1
            z = abs(mean_a - mean_b) / std_a
            if z > 1.5:
                diffs.append(f"mean shift: {mean_a:.2f} → {mean_b:.2f} (z={z:.1f})")
        if diffs:
            changes.append({"col": col, "diffs": diffs})

    return {
        "name_a": name_a,
        "name_b": name_b,
        "added": added,
        "removed": removed,
        "changes": changes,
        "stable": [c for c in common if not any(ch["col"] == c for ch in changes)],
    }