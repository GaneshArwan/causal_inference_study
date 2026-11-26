"""Causal inference estimators, covariate balance diagnostics, and sensitivity analysis."""

from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.guardrails import (
    COVARIATE_COLUMNS,
    check_covariate_balance,
    check_positivity_overlap,
    check_post_trim_groups,
    require_covariates,
    validate_dataframe,
)
from src.schemas import (
    CovariateBalanceSummary,
    EstimatorResult,
    PropensityModelMetrics,
    SensitivityResult,
    StudyConfig,
    StudyResults,
)

Z_95 = 1.959963984540054


def _logit(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _summarize_bootstrap(
    method: str,
    estimate: float,
    boot_samples: np.ndarray,
    n_treated: int,
    n_control: int,
) -> EstimatorResult:
    se = float(np.std(boot_samples, ddof=1))
    ci_low, ci_high = (float(x) for x in np.percentile(boot_samples, [2.5, 97.5]))
    return EstimatorResult(
        method=method,
        estimate=estimate,
        se=se,
        ci_low=ci_low,
        ci_high=ci_high,
        n_treated=n_treated,
        n_control=n_control,
    )


def compute_smd(
    df: pd.DataFrame,
    covariates: Sequence[str] = COVARIATE_COLUMNS,
    treat_col: str = "treat",
) -> Dict[str, float]:
    """Compute Standardized Mean Difference (treated - control) per covariate."""
    treated = df[df[treat_col] == 1]
    control = df[df[treat_col] == 0]
    if len(treated) < 2 or len(control) < 2:
        raise ValueError("Need at least 2 observations per group to compute SMD.")

    smds: Dict[str, float] = {}
    for col in covariates:
        m1 = float(treated[col].mean())
        m0 = float(control[col].mean())
        v1 = float(treated[col].var(ddof=1))
        v0 = float(control[col].var(ddof=1))
        pooled_sd = float(np.sqrt(0.5 * (v1 + v0)))
        smds[col] = 0.0 if (not np.isfinite(pooled_sd) or pooled_sd < 1e-12) else (m1 - m0) / pooled_sd
    return smds


def estimate_naive_ate(df: pd.DataFrame) -> EstimatorResult:
    """Unadjusted difference in post-period earnings (re78) with Welch SE."""
    y1 = df.loc[df["treat"] == 1, "re78"].to_numpy(dtype=float)
    y0 = df.loc[df["treat"] == 0, "re78"].to_numpy(dtype=float)
    diff = float(y1.mean() - y0.mean())
    se = float(np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0)))
    return EstimatorResult(
        method="Naive Difference-in-Means",
        estimate=diff,
        se=se,
        ci_low=diff - Z_95 * se,
        ci_high=diff + Z_95 * se,
        n_treated=len(y1),
        n_control=len(y0),
    )


def fit_propensity_scores(
    df: pd.DataFrame,
    covariates: Sequence[str] = COVARIATE_COLUMNS,
    trim_bounds: Tuple[float, float] = (0.01, 0.99),
    seed: int = 42,
) -> Tuple[np.ndarray, PropensityModelMetrics, Tuple[float, float]]:
    """Fit regularized propensity score model and evaluate 5-fold CV diagnostics."""
    covs = require_covariates(covariates, df.columns.tolist())
    x = df.loc[:, list(covs)].to_numpy(dtype=float)
    t_vec = df["treat"].to_numpy(dtype=int)

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=seed)),
        ]
    )

    n_splits = min(5, int((t_vec == 1).sum()), int((t_vec == 0).sum()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_probs = cross_val_predict(pipe, x, t_vec, cv=cv, method="predict_proba")[:, 1]

    auc = float(roc_auc_score(t_vec, oof_probs))
    threshold = float(t_vec.mean())
    preds = (oof_probs >= threshold).astype(int)
    precision = float(precision_score(t_vec, preds, zero_division=0))
    recall = float(recall_score(t_vec, preds, zero_division=0))

    pipe.fit(x, t_vec)
    ps = pipe.predict_proba(x)[:, 1]
    overlap = check_positivity_overlap(ps, t_vec, trim_bounds=trim_bounds)
    support_bounds = (float(overlap["support_low"]), float(overlap["support_high"]))

    metrics = PropensityModelMetrics(
        cv_auc_roc=auc,
        cv_precision=precision,
        cv_recall=recall,
        common_support_fraction=float(overlap["overlap_fraction"]),
        treated_in_support=int(overlap["treated_in_support"]),
        control_in_support=int(overlap["control_in_support"]),
    )
    return ps, metrics, support_bounds


def _match_on_logit_ps(
    df: pd.DataFrame,
    ps: np.ndarray,
    caliper_sd: float,
    min_pairs: int = 10,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Perform 1:1 optimal nearest-neighbor matching WITHOUT replacement within logit PS caliper.

    Matching without replacement via linear_sum_assignment minimizes total logit-PS
    distance and yields independent pairs so pairs bootstrap and Rosenbaum sign tests
    remain statistically valid.
    """
    logit_ps = _logit(ps)
    caliper = caliper_sd * float(np.std(logit_ps, ddof=1))

    treated_idx = np.flatnonzero(df["treat"].to_numpy() == 1)
    control_idx = np.flatnonzero(df["treat"].to_numpy() == 0)

    dist_matrix = np.abs(logit_ps[treated_idx, None] - logit_ps[control_idx][None, :])
    penalty = max(caliper * 100.0, 1e6)
    cost_matrix = np.where(dist_matrix <= caliper, dist_matrix, penalty)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    within = dist_matrix[row_ind, col_ind] <= caliper
    matched_treated = treated_idx[row_ind[within]]
    matched_control = control_idx[col_ind[within]]

    if len(matched_treated) < min_pairs:
        raise ValueError(
            f"Too few matched pairs ({len(matched_treated)}) within caliper {caliper:.4f}."
        )

    matched_df = pd.concat(
        [df.iloc[matched_treated], df.iloc[matched_control]],
        axis=0,
        ignore_index=True,
    )
    return matched_df, matched_treated, matched_control


def estimate_psm_att(
    df: pd.DataFrame,
    ps: np.ndarray,
    covariates: Sequence[str] = COVARIATE_COLUMNS,
    caliper_sd: float = 0.2,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> Tuple[EstimatorResult, pd.DataFrame, Dict[str, float], np.ndarray]:
    """Estimate ATT via 1:1 without-replacement Propensity Score Matching."""
    if n_bootstrap < 2:
        raise ValueError("n_bootstrap must be at least 2.")

    matched_df, m_t, m_c = _match_on_logit_ps(df, ps, caliper_sd=caliper_sd)
    pair_diffs = df.iloc[m_t]["re78"].to_numpy(dtype=float) - df.iloc[m_c]["re78"].to_numpy(dtype=float)
    att = float(np.mean(pair_diffs))
    smd_after = compute_smd(matched_df, covariates=covariates)

    rng = np.random.default_rng(seed)
    n_pairs = len(pair_diffs)
    boot_ests = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        sample_idx = rng.integers(0, n_pairs, size=n_pairs)
        boot_ests[b] = float(np.mean(pair_diffs[sample_idx]))

    result = _summarize_bootstrap(
        method="Propensity Score Matching (ATT)",
        estimate=att,
        boot_samples=boot_ests,
        n_treated=len(m_t),
        n_control=len(m_c),
    )
    return result, matched_df, smd_after, pair_diffs


def _hajek_ipw_att(y: np.ndarray, treat: np.ndarray, ps: np.ndarray) -> float:
    t_mask = treat == 1
    c_mask = treat == 0
    w_c = ps[c_mask] / (1.0 - ps[c_mask])
    mu_t = float(np.mean(y[t_mask]))
    mu_c = float(np.sum(w_c * y[c_mask]) / np.sum(w_c))
    return mu_t - mu_c


def estimate_ipw_att(
    df: pd.DataFrame,
    ps: np.ndarray,
    support_bounds: Tuple[float, float] = (0.01, 0.99),
    n_bootstrap: int = 200,
    seed: int = 42,
) -> EstimatorResult:
    """Estimate ATT via Hajek-normalized IPW with stratified bootstrap CI."""
    if n_bootstrap < 2:
        raise ValueError("n_bootstrap must be at least 2.")

    low, high = support_bounds
    keep = (ps >= low) & (ps <= high)
    y = df.loc[keep, "re78"].to_numpy(dtype=float)
    t_vec = df.loc[keep, "treat"].to_numpy(dtype=int)
    ps_kept = ps[keep]
    check_post_trim_groups(t_vec, min_group_size=5)

    att = _hajek_ipw_att(y, t_vec, ps_kept)
    t_idx = np.flatnonzero(t_vec == 1)
    c_idx = np.flatnonzero(t_vec == 0)

    rng = np.random.default_rng(seed)
    boot = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        draw = np.concatenate(
            [
                rng.choice(t_idx, size=len(t_idx), replace=True),
                rng.choice(c_idx, size=len(c_idx), replace=True),
            ]
        )
        boot[b] = _hajek_ipw_att(y[draw], t_vec[draw], ps_kept[draw])

    return _summarize_bootstrap(
        method="Inverse Probability Weighting (ATT)",
        estimate=att,
        boot_samples=boot,
        n_treated=len(t_idx),
        n_control=len(c_idx),
    )


def estimate_aipw_att(
    df: pd.DataFrame,
    ps: np.ndarray,
    covariates: Sequence[str] = COVARIATE_COLUMNS,
    support_bounds: Tuple[float, float] = (0.01, 0.99),
    seed: int = 42,
) -> EstimatorResult:
    """Doubly Robust Augmented IPW estimator for the ATT using cross-fitted Ridge."""
    low, high = support_bounds
    keep = (ps >= low) & (ps <= high)
    sub = df.loc[keep].reset_index(drop=True)
    ps_kept = ps[keep]

    x = sub.loc[:, list(covariates)].to_numpy(dtype=float)
    y = sub["re78"].to_numpy(dtype=float)
    t_vec = sub["treat"].to_numpy(dtype=int)
    check_post_trim_groups(t_vec, min_group_size=5)

    n_splits = min(5, int((t_vec == 1).sum()), int((t_vec == 0).sum()))
    mu0_hat = np.zeros(len(sub), dtype=float)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, val_idx in skf.split(x, t_vec):
        ctrl_train = train_idx[t_vec[train_idx] == 0]
        model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
        model.fit(x[ctrl_train], y[ctrl_train])
        mu0_hat[val_idx] = model.predict(x[val_idx])

    p_treat = float(np.mean(t_vec))
    odds = ps_kept / (1.0 - ps_kept)
    psi = ((t_vec / p_treat) - ((1 - t_vec) * odds / p_treat)) * (y - mu0_hat)
    att = float(np.mean(psi))
    se = float(np.std(psi, ddof=1) / np.sqrt(len(psi)))

    return EstimatorResult(
        method="Doubly Robust AIPW (ATT)",
        estimate=att,
        se=se,
        ci_low=att - Z_95 * se,
        ci_high=att + Z_95 * se,
        n_treated=int((t_vec == 1).sum()),
        n_control=int((t_vec == 0).sum()),
    )


def estimate_did(df: pd.DataFrame) -> EstimatorResult:
    """Two-period Difference-in-Differences on earnings change (re78 - re75)."""
    delta_t = (df.loc[df["treat"] == 1, "re78"] - df.loc[df["treat"] == 1, "re75"]).to_numpy(dtype=float)
    delta_c = (df.loc[df["treat"] == 0, "re78"] - df.loc[df["treat"] == 0, "re75"]).to_numpy(dtype=float)
    did = float(np.mean(delta_t) - np.mean(delta_c))
    se = float(np.sqrt(delta_t.var(ddof=1) / len(delta_t) + delta_c.var(ddof=1) / len(delta_c)))
    return EstimatorResult(
        method="Difference-in-Differences (DiD)",
        estimate=did,
        se=se,
        ci_low=did - Z_95 * se,
        ci_high=did + Z_95 * se,
        n_treated=len(delta_t),
        n_control=len(delta_c),
    )


def run_sensitivity_analysis(
    df: pd.DataFrame,
    ps: np.ndarray,
    psm_result: EstimatorResult,
    pair_diffs: np.ndarray,
    caliper_sd: float = 0.2,
    seed: int = 42,
) -> SensitivityResult:
    """Compute E-value, control-arm placebo test, and Rosenbaum critical Gamma."""
    ctrl_sd = float(df.loc[df["treat"] == 0, "re78"].std(ddof=1))
    d_point = abs(psm_result.estimate) / max(ctrl_sd, 1e-6)

    # VanderWeele & Ding (2017): E-value for CI is 1.0 whenever CI straddles the null
    if psm_result.ci_low <= 0.0 <= psm_result.ci_high:
        d_ci = 0.0
    else:
        bound = psm_result.ci_low if psm_result.estimate >= 0 else abs(psm_result.ci_high)
        d_ci = max(0.0, bound / max(ctrl_sd, 1e-6))

    rr_point = float(np.exp(0.91 * d_point))
    rr_ci = float(np.exp(0.91 * d_ci))
    e_val_point = float(rr_point + np.sqrt(max(0.0, rr_point * (rr_point - 1.0))))
    e_val_ci = 1.0 if rr_ci <= 1.0 else float(rr_ci + np.sqrt(rr_ci * (rr_ci - 1.0)))

    # Placebo treatment test on control units only
    ctrl_df = df[df["treat"] == 0].copy().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    n_placebo_t = min(int((df["treat"] == 1).sum()), len(ctrl_df) // 2)
    n_placebo_t = max(10, min(n_placebo_t, len(ctrl_df) - 10))
    placebo_t = np.zeros(len(ctrl_df), dtype=int)
    placebo_t[rng.choice(len(ctrl_df), size=n_placebo_t, replace=False)] = 1
    ctrl_df["treat"] = placebo_t
    placebo_ps = ps[df["treat"].to_numpy() == 0]
    _, p_t, p_c = _match_on_logit_ps(
        ctrl_df,
        placebo_ps,
        caliper_sd=max(caliper_sd, 1.5),
        min_pairs=min(10, n_placebo_t),
    )
    placebo_diffs = ctrl_df.iloc[p_t]["re78"].to_numpy(dtype=float) - ctrl_df.iloc[p_c]["re78"].to_numpy(dtype=float)
    placebo_est = float(np.mean(placebo_diffs))
    if len(placebo_diffs) < 2 or np.std(placebo_diffs, ddof=1) < 1e-12:
        placebo_p = 1.0
    else:
        _, raw_p = stats.ttest_1samp(placebo_diffs, 0.0)
        placebo_p = 1.0 if not np.isfinite(raw_p) else float(raw_p)

    # Rosenbaum sensitivity Gamma on independent without-replacement matched pairs
    nonzero = pair_diffs[pair_diffs != 0]
    n_nonzero = len(nonzero)
    if n_nonzero == 0:
        gamma_critical = 1.0
    else:
        n_favorable = int(np.sum(nonzero > 0)) if psm_result.estimate >= 0 else int(np.sum(nonzero < 0))
        gamma_critical = 5.0
        for gamma in np.arange(1.0, 5.05, 0.05):
            p_upper = gamma / (1.0 + gamma)
            p_val = float(stats.binomtest(n_favorable, n_nonzero, p=p_upper, alternative="greater").pvalue)
            if p_val > 0.05:
                gamma_critical = float(round(gamma, 2))
                break

    return SensitivityResult(
        e_value_point=round(e_val_point, 3),
        e_value_ci=round(e_val_ci, 3),
        placebo_effect=round(placebo_est, 2),
        placebo_p_value=round(placebo_p, 4),
        gamma_critical=gamma_critical,
    )


def run_full_study(
    df: pd.DataFrame,
    config: StudyConfig | None = None,
    true_att_benchmark: float = 1800.0,
) -> Tuple[StudyResults, np.ndarray, pd.DataFrame]:
    """Execute the end-to-end causal inference pipeline on a validated DataFrame."""
    cfg = config or StudyConfig()
    clean_df = validate_dataframe(df)

    smd_before = compute_smd(clean_df, covariates=COVARIATE_COLUMNS)
    naive = estimate_naive_ate(clean_df)
    ps, ps_metrics, support_bounds = fit_propensity_scores(
        clean_df,
        covariates=COVARIATE_COLUMNS,
        trim_bounds=cfg.trim_bounds,
        seed=cfg.seed,
    )
    psm, matched_df, smd_after, pair_diffs = estimate_psm_att(
        clean_df,
        ps,
        covariates=COVARIATE_COLUMNS,
        caliper_sd=cfg.caliper_sd_multiplier,
        n_bootstrap=cfg.n_bootstrap,
        seed=cfg.seed,
    )
    ipw = estimate_ipw_att(
        clean_df,
        ps,
        support_bounds=support_bounds,
        n_bootstrap=cfg.n_bootstrap,
        seed=cfg.seed,
    )
    aipw = estimate_aipw_att(
        clean_df,
        ps,
        covariates=COVARIATE_COLUMNS,
        support_bounds=support_bounds,
        seed=cfg.seed,
    )
    did = estimate_did(clean_df)
    sensitivity = run_sensitivity_analysis(
        clean_df,
        ps,
        psm,
        pair_diffs=pair_diffs,
        caliper_sd=cfg.caliper_sd_multiplier,
        seed=cfg.seed,
    )

    balance = CovariateBalanceSummary(
        smd_before={k: round(v, 4) for k, v in smd_before.items()},
        smd_after={k: round(v, 4) for k, v in smd_after.items()},
        max_abs_smd_before=round(max(abs(v) for v in smd_before.values()), 4),
        max_abs_smd_after=round(max(abs(v) for v in smd_after.values()), 4),
        balanced=check_covariate_balance(smd_after, threshold=cfg.max_smd_threshold),
    )

    results = StudyResults(
        config=cfg,
        n_total=len(clean_df),
        n_treated=int((clean_df["treat"] == 1).sum()),
        n_control=int((clean_df["treat"] == 0).sum()),
        true_att_benchmark=true_att_benchmark,
        propensity_metrics=ps_metrics,
        covariate_balance=balance,
        estimators=[naive, psm, ipw, aipw, did],
        sensitivity=sensitivity,
    )
    return results, ps, matched_df
