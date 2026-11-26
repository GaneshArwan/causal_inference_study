"""Pydantic v2 schemas for configuration, data validation, and study outputs."""

from typing import Dict, List, Literal, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StudyConfig(BaseModel):
    """Configuration for the causal inference study pipeline."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    n_samples: int = Field(default=2675, ge=200, le=50000)
    caliper_sd_multiplier: float = Field(default=0.2, ge=0.01, le=2.0)
    n_bootstrap: int = Field(default=200, ge=50, le=2000)
    trim_bounds: Tuple[float, float] = Field(default=(0.01, 0.99))
    locale: Literal["en", "id"] = Field(default="en")
    max_smd_threshold: float = Field(default=0.1, gt=0.0, le=0.5)

    @model_validator(mode="after")
    def _validate_trim_bounds(self) -> "StudyConfig":
        low, high = self.trim_bounds
        if not (0.0 < low < high < 1.0):
            raise ValueError("trim_bounds must satisfy 0 < low < high < 1")
        return self


class ObservationRecord(BaseModel):
    """Schema for a single LaLonde-style observational study unit."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    treat: Literal[0, 1]
    age: int = Field(ge=16, le=80)
    education: int = Field(ge=0, le=25)
    black: Literal[0, 1]
    hispanic: Literal[0, 1]
    married: Literal[0, 1]
    nodegree: Literal[0, 1]
    re74: float = Field(ge=0.0)
    re75: float = Field(ge=0.0)
    re78: float = Field(ge=0.0)


class PropensityModelMetrics(BaseModel):
    """Cross-validated classification diagnostics for the propensity model."""

    model_config = ConfigDict(allow_inf_nan=False)

    cv_auc_roc: float = Field(ge=0.0, le=1.0)
    cv_precision: float = Field(ge=0.0, le=1.0)
    cv_recall: float = Field(ge=0.0, le=1.0)
    common_support_fraction: float = Field(ge=0.0, le=1.0)
    treated_in_support: int = Field(ge=0)
    control_in_support: int = Field(ge=0)


class EstimatorResult(BaseModel):
    """Point estimate, uncertainty bounds, and sample sizes for an estimator."""

    model_config = ConfigDict(allow_inf_nan=False)

    method: str
    estimate: float
    se: float = Field(ge=0.0)
    ci_low: float
    ci_high: float
    n_treated: int = Field(ge=0)
    n_control: int = Field(ge=0)


class CovariateBalanceSummary(BaseModel):
    """Standardized mean differences before and after matching."""

    model_config = ConfigDict(allow_inf_nan=False)

    smd_before: Dict[str, float]
    smd_after: Dict[str, float]
    max_abs_smd_before: float = Field(ge=0.0)
    max_abs_smd_after: float = Field(ge=0.0)
    balanced: bool


class SensitivityResult(BaseModel):
    """Robustness diagnostics against unmeasured confounding and placebo bias."""

    model_config = ConfigDict(allow_inf_nan=False)

    e_value_point: float = Field(ge=1.0)
    e_value_ci: float = Field(ge=1.0)
    placebo_effect: float
    placebo_p_value: float = Field(ge=0.0, le=1.0)
    gamma_critical: float = Field(ge=1.0)


class StudyResults(BaseModel):
    """Complete serialized output of the causal inference study."""

    model_config = ConfigDict(allow_inf_nan=False)

    config: StudyConfig
    n_total: int = Field(ge=1)
    n_treated: int = Field(ge=1)
    n_control: int = Field(ge=1)
    true_att_benchmark: float
    propensity_metrics: PropensityModelMetrics
    covariate_balance: CovariateBalanceSummary
    estimators: List[EstimatorResult]
    sensitivity: SensitivityResult
