"""Runtime safety, input validation, and statistical assumption guardrails."""

from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import pandas as pd

from src.schemas import ObservationRecord

REQUIRED_COLUMNS: Tuple[str, ...] = (
    "treat",
    "age",
    "education",
    "black",
    "hispanic",
    "married",
    "nodegree",
    "re74",
    "re75",
    "re78",
)

COVARIATE_COLUMNS: Tuple[str, ...] = (
    "age",
    "education",
    "black",
    "hispanic",
    "married",
    "nodegree",
    "re74",
    "re75",
)

BINARY_COLUMNS: Tuple[str, ...] = ("treat", "black", "hispanic", "married", "nodegree")
INTEGER_COLUMNS: Tuple[str, ...] = ("treat", "age", "education", "black", "hispanic", "married", "nodegree")
NON_NEGATIVE_COLUMNS: Tuple[str, ...] = ("re74", "re75", "re78")


def validate_safe_path(target_path: Path, base_dir: Path) -> Path:
    """Resolve target_path and ensure it stays strictly inside base_dir."""
    resolved_base = base_dir.resolve()
    resolved_target = target_path.resolve()
    try:
        resolved_target.relative_to(resolved_base)
    except ValueError as exc:
        raise ValueError(
            f"Unsafe path '{resolved_target}' escapes base directory '{resolved_base}'."
        ) from exc
    return resolved_target


def validate_dataframe(
    df: pd.DataFrame,
    min_group_size: int = 20,
    sample_validation_limit: int = 200,
) -> pd.DataFrame:
    """Validate dataset schema, finiteness, full-column domain bounds, and record types."""
    if df.empty:
        raise ValueError("Input dataset is empty.")

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    numeric_df = df.loc[:, list(REQUIRED_COLUMNS)]
    if numeric_df.isna().any().any():
        raise ValueError("Dataset contains NaN values in required columns.")

    values = numeric_df.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Dataset contains non-finite (Inf/-Inf) values.")

    # Full-column vectorized domain validation matching ObservationRecord constraints
    for col in INTEGER_COLUMNS:
        col_vals = numeric_df[col].to_numpy(dtype=float)
        if not np.all(np.floor(col_vals) == col_vals):
            raise ValueError(f"Column '{col}' must contain integer values.")

    for col in BINARY_COLUMNS:
        if not numeric_df[col].isin([0, 1]).all():
            raise ValueError(f"Column '{col}' must contain only binary values 0 and 1.")

    if not ((numeric_df["age"] >= 16) & (numeric_df["age"] <= 80)).all():
        raise ValueError("Column 'age' must be within [16, 80].")
    if not ((numeric_df["education"] >= 0) & (numeric_df["education"] <= 25)).all():
        raise ValueError("Column 'education' must be within [0, 25].")
    for col in NON_NEGATIVE_COLUMNS:
        if (numeric_df[col] < 0.0).any():
            raise ValueError(f"Column '{col}' must be non-negative.")

    n_treated = int((numeric_df["treat"] == 1).sum())
    n_control = int((numeric_df["treat"] == 0).sum())
    if n_treated == 0 or n_control == 0:
        raise ValueError("Column 'treat' must contain both binary values 0 and 1.")
    if n_treated < min_group_size or n_control < min_group_size:
        raise ValueError(
            f"Insufficient group size (treated={n_treated}, control={n_control}; "
            f"minimum={min_group_size})."
        )

    # Validate representative head/tail records via ObservationRecord for strict Pydantic parity
    half = max(1, sample_validation_limit // 2)
    sample_indices = sorted(set(range(min(len(df), half))) | set(range(max(0, len(df) - half), len(df))))
    for row in numeric_df.iloc[sample_indices].to_dict(orient="records"):
        ObservationRecord.model_validate(row)

    return df.copy()


def check_positivity_overlap(
    propensity_scores: np.ndarray,
    treat: np.ndarray,
    trim_bounds: Tuple[float, float] = (0.01, 0.99),
    min_overlap_fraction: float = 0.5,
) -> Dict[str, float | int]:
    """Verify common support / positivity overlap between treated and control units."""
    if len(propensity_scores) != len(treat):
        raise ValueError("propensity_scores and treat must have equal length.")

    low, high = trim_bounds
    ps_treated = propensity_scores[treat == 1]
    ps_control = propensity_scores[treat == 0]
    if len(ps_treated) == 0 or len(ps_control) == 0:
        raise ValueError("Both treated and control observations are required.")

    support_low = max(low, float(ps_treated.min()), float(ps_control.min()))
    support_high = min(high, float(ps_treated.max()), float(ps_control.max()))
    if support_low >= support_high:
        raise ValueError("Positivity violation: zero common support between groups.")

    in_support = (propensity_scores >= support_low) & (propensity_scores <= support_high)
    overlap_fraction = float(in_support.mean())
    if overlap_fraction < min_overlap_fraction:
        raise ValueError(
            f"Positivity overlap too low ({overlap_fraction:.2%} < {min_overlap_fraction:.2%})."
        )

    return {
        "overlap_fraction": overlap_fraction,
        "treated_in_support": int((in_support & (treat == 1)).sum()),
        "control_in_support": int((in_support & (treat == 0)).sum()),
        "support_low": support_low,
        "support_high": support_high,
    }


def check_post_trim_groups(treat: np.ndarray, min_group_size: int = 5) -> None:
    """Verify both treatment arms retain sufficient observations after propensity trimming."""
    n_t = int((treat == 1).sum())
    n_c = int((treat == 0).sum())
    if n_t < min_group_size or n_c < min_group_size:
        raise ValueError(
            f"Insufficient observations after trimming (treated={n_t}, control={n_c}; "
            f"minimum={min_group_size})."
        )


def check_covariate_balance(
    smd_after: Dict[str, float],
    threshold: float = 0.1,
) -> bool:
    """Return True when all post-matching absolute SMDs meet the balance threshold."""
    if not smd_after:
        raise ValueError("smd_after cannot be empty.")
    return all(abs(val) <= threshold for val in smd_after.values())


def require_covariates(columns: Iterable[str], available: Sequence[str]) -> Tuple[str, ...]:
    """Ensure requested covariates exist in the dataset."""
    col_tuple = tuple(columns)
    missing = [col for col in col_tuple if col not in available]
    if missing:
        raise ValueError(f"Requested covariates missing from dataset: {missing}")
    return col_tuple
