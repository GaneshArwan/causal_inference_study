"""Unit and integration tests for the causal inference study pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.causal_engine import (
    compute_smd,
    estimate_aipw_att,
    estimate_ipw_att,
    run_full_study,
    run_sensitivity_analysis,
)
from src.data_generator import PROJECT_ROOT, TRUE_ATT_BENCHMARK, generate_lalonde_dataset
from src.guardrails import (
    check_covariate_balance,
    check_positivity_overlap,
    validate_dataframe,
    validate_safe_path,
)
from src.i18n import fmt_num, fmt_pct, fmt_usd, t
from src.schemas import EstimatorResult, ObservationRecord, StudyConfig
from src.train import render_markdown_report, run_pipeline


def test_study_config_and_observation_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="trim_bounds"):
        StudyConfig(trim_bounds=(0.9, 0.1))

    with pytest.raises(ValueError):
        ObservationRecord(
            treat=1, age=30, education=12, black=0, hispanic=0,
            married=1, nodegree=0, re74=1000.0, re75=1200.0, re78=float("inf"),
        )


def test_validate_safe_path_blocks_traversal() -> None:
    with pytest.raises(ValueError, match="Unsafe path"):
        validate_safe_path(PROJECT_ROOT.parent / "outside.csv", PROJECT_ROOT)


def test_validate_dataframe_enforces_full_column_bounds_beyond_row_200() -> None:
    df = generate_lalonde_dataset(n_samples=300, seed=42)
    with pytest.raises(ValueError, match="missing required columns"):
        validate_dataframe(df.drop(columns=["re78"]))

    df_nan = df.copy()
    df_nan.loc[0, "re75"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        validate_dataframe(df_nan)

    df_late_neg = df.copy()
    df_late_neg.loc[250, "re78"] = -50.0
    with pytest.raises(ValueError, match="non-negative"):
        validate_dataframe(df_late_neg)

    df_late_frac = df.copy()
    df_late_frac["age"] = df_late_frac["age"].astype(float)
    df_late_frac.loc[260, "age"] = 30.5
    with pytest.raises(ValueError, match="integer"):
        validate_dataframe(df_late_frac)


def test_validate_dataframe_rejects_inf_and_single_treatment() -> None:
    df = generate_lalonde_dataset(n_samples=300, seed=42)
    df_inf = df.copy()
    df_inf.loc[0, "re78"] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        validate_dataframe(df_inf)

    df_one_arm = df.copy()
    df_one_arm["treat"] = 0
    with pytest.raises(ValueError, match="both binary values"):
        validate_dataframe(df_one_arm)


def test_positivity_and_post_trim_guardrails() -> None:
    ps = np.array([0.05] * 30 + [0.95] * 30)
    treat = np.array([0] * 30 + [1] * 30)
    with pytest.raises(ValueError, match="zero common support"):
        check_positivity_overlap(ps, treat)

    df = generate_lalonde_dataset(n_samples=250, seed=42)
    narrow_ps = np.full(len(df), 0.5)
    narrow_ps[df["treat"] == 1] = 0.999
    with pytest.raises(ValueError, match="Insufficient observations after trimming"):
        estimate_ipw_att(df, narrow_ps, support_bounds=(0.1, 0.9), n_bootstrap=10)
    with pytest.raises(ValueError, match="Insufficient observations after trimming"):
        estimate_aipw_att(df, narrow_ps, support_bounds=(0.1, 0.9))


def test_covariate_balance_and_smd_small_group_guardrail() -> None:
    assert check_covariate_balance({"age": 0.05, "re75": -0.09}, threshold=0.1) is True
    assert check_covariate_balance({"age": 0.15, "re75": -0.02}, threshold=0.1) is False

    tiny_df = generate_lalonde_dataset(n_samples=250, seed=42).iloc[:20].copy()
    tiny_df["treat"] = 0
    tiny_df.iloc[0, tiny_df.columns.get_loc("treat")] = 1
    with pytest.raises(ValueError, match="at least 2 observations"):
        compute_smd(tiny_df)


def test_data_generator_is_deterministic() -> None:
    df_a = generate_lalonde_dataset(n_samples=500, seed=123)
    df_b = generate_lalonde_dataset(n_samples=500, seed=123)
    pd.testing.assert_frame_equal(df_a, df_b)


def test_sensitivity_handles_null_straddling_ci_small_controls_and_zero_diffs() -> None:
    df = generate_lalonde_dataset(n_samples=250, seed=42)
    ctrl_subset = df[df["treat"] == 0].iloc[:22]
    treat_subset = df[df["treat"] == 1].iloc[:20]
    small_df = pd.concat([treat_subset, ctrl_subset], ignore_index=True)
    ps = np.linspace(0.2, 0.8, len(small_df))

    null_ci_est = EstimatorResult(
        method="Propensity Score Matching (ATT)",
        estimate=500.0,
        se=400.0,
        ci_low=-284.0,
        ci_high=1284.0,
        n_treated=20,
        n_control=20,
    )
    sens = run_sensitivity_analysis(
        small_df,
        ps,
        null_ci_est,
        pair_diffs=np.zeros(20, dtype=float),
        caliper_sd=0.5,
        seed=42,
    )
    assert sens.e_value_ci == 1.0
    assert sens.gamma_critical == 1.0
    assert 0.0 <= sens.placebo_p_value <= 1.0


def test_full_study_recovers_positive_att_and_balances_covariates() -> None:
    df = generate_lalonde_dataset(n_samples=2675, seed=42)
    cfg = StudyConfig(seed=42, n_bootstrap=100)
    results, _, _ = run_full_study(df, config=cfg, true_att_benchmark=TRUE_ATT_BENCHMARK)

    est_by_name = {e.method: e for e in results.estimators}
    naive = est_by_name["Naive Difference-in-Means"]
    psm = est_by_name["Propensity Score Matching (ATT)"]
    ipw = est_by_name["Inverse Probability Weighting (ATT)"]
    aipw = est_by_name["Doubly Robust AIPW (ATT)"]
    did = est_by_name["Difference-in-Differences (DiD)"]

    assert naive.estimate < -2000.0
    for causal_est in (psm, ipw, aipw, did):
        assert 1000.0 <= causal_est.estimate <= 2600.0
        assert causal_est.ci_low > 0.0

    assert results.covariate_balance.max_abs_smd_before > 0.3
    assert results.covariate_balance.max_abs_smd_after <= 0.10
    assert results.covariate_balance.balanced is True

    assert results.sensitivity.e_value_point > 1.5
    assert results.sensitivity.placebo_p_value > 0.05
    assert results.sensitivity.gamma_critical >= 1.5


def test_i18n_translations_and_babel_formatting() -> None:
    assert "Causal" in t("app_title", "en")
    assert "Kausal" in t("app_title", "id")
    assert t("app_title", "fr") == t("app_title", "en")
    assert "1,800" in fmt_usd(1800.0, "en")
    assert "0.850" in fmt_num(0.85, "en", 3)
    assert "%" in fmt_pct(0.95, "en")


def test_end_to_end_pipeline_writes_artifacts(tmp_path: Path) -> None:
    cfg = StudyConfig(seed=42, n_bootstrap=60, locale="id")
    results, _ = run_pipeline(cfg, overwrite_data=True, base_dir=tmp_path)

    reports_dir = tmp_path / "reports"
    figures_dir = reports_dir / "figures"
    assert (tmp_path / "data" / "lalonde_study.csv").exists()
    assert (reports_dir / "results.json").exists()
    assert (reports_dir / "causal_report.md").exists()
    for name in ("dag.png", "propensity_overlap.png", "covariate_balance.png", "effect_comparison.png"):
        assert (figures_dir / name).exists()

    md_id = render_markdown_report(results, locale="id")
    assert "Ringkasan Eksekutif" in md_id
    assert "Keterbatasan" in md_id
    assert "Causal estimates rely on" not in md_id
