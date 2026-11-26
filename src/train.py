"""CLI entrypoint to run the causal inference study, generate figures, and write reports."""

import argparse
import sys
from pathlib import Path
from typing import Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.causal_engine import run_full_study
from src.data_generator import PROJECT_ROOT, load_or_generate_dataset
from src.guardrails import validate_safe_path
from src.i18n import fmt_num, fmt_pct, fmt_usd, t
from src.schemas import StudyConfig, StudyResults

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def _save_or_return(fig: Figure, output_path: Path | None) -> Figure:
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
    return fig


def build_dag_figure(output_path: Path | None = None) -> Figure:
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=150)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    nodes = {
        "Confounders": (5.0, 4.9, "#E8EEF5", "#2B4C7E"),
        "Treatment": (1.8, 1.5, "#FCE8D5", "#B45309"),
        "Outcome": (8.2, 1.5, "#DCFCE7", "#166534"),
        "Unobserved": (5.0, 0.5, "#F3F4F6", "#6B7280"),
    }
    labels = {
        "Confounders": "Baseline Covariates (X)\nAge, Education, Prior Earnings",
        "Treatment": "Job Training (T)\nProgram Enrollment",
        "Outcome": "Post Earnings (Y)\n1978 Real Earnings",
        "Unobserved": "Unobserved U\n(Tested via E-value & Gamma)",
    }

    for key, (x, y, fc, ec) in nodes.items():
        box_style = "round,pad=0.45,rounding_size=0.2"
        ls = "--" if key == "Unobserved" else "-"
        ax.text(
            x,
            y,
            labels[key],
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold" if key != "Unobserved" else "normal",
            bbox=dict(boxstyle=box_style, facecolor=fc, edgecolor=ec, linewidth=1.6, linestyle=ls),
        )

    arrow_kw = dict(arrowstyle="-|>", lw=1.8, mutation_scale=15, color="#374151")
    ax.annotate("", xy=(2.2, 2.15), xytext=(4.2, 4.25), arrowprops=arrow_kw)
    ax.annotate("", xy=(7.8, 2.15), xytext=(5.8, 4.25), arrowprops=arrow_kw)
    ax.annotate("", xy=(6.5, 1.5), xytext=(3.5, 1.5), arrowprops=dict(arrowstyle="-|>", lw=2.3, mutation_scale=17, color="#166534"))
    ax.text(5.0, 1.82, "Causal Effect (ATT)", ha="center", fontsize=9.5, fontweight="bold", color="#166534")

    dashed_kw = dict(arrowstyle="-|>", lw=1.2, ls="--", mutation_scale=12, color="#9CA3AF")
    ax.annotate("", xy=(2.9, 1.15), xytext=(4.0, 0.7), arrowprops=dashed_kw)
    ax.annotate("", xy=(7.1, 1.15), xytext=(6.0, 0.7), arrowprops=dashed_kw)

    ax.set_title("Causal Directed Acyclic Graph (DAG): Selection Bias vs. Treatment Effect", fontsize=11.5, fontweight="bold", pad=10)
    return _save_or_return(fig, output_path)


def build_propensity_overlap_figure(df: pd.DataFrame, ps: np.ndarray, output_path: Path | None = None) -> Figure:
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    t_mask = df["treat"].to_numpy() == 1
    sns.kdeplot(ps[~t_mask], label="Control (T=0)", fill=True, alpha=0.35, color="#2563EB", clip=(0, 1), ax=ax)
    sns.kdeplot(ps[t_mask], label="Treated (T=1)", fill=True, alpha=0.35, color="#D97706", clip=(0, 1), ax=ax)
    ax.set_xlabel("Estimated Propensity Score P(T=1 | X)")
    ax.set_ylabel("Density")
    ax.set_title("Propensity Score Overlap (Common Support)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    return _save_or_return(fig, output_path)


def build_covariate_balance_figure(results: StudyResults, output_path: Path | None = None) -> Figure:
    bal = results.covariate_balance
    covs = list(bal.smd_before.keys())
    before = [abs(bal.smd_before[c]) for c in covs]
    after = [abs(bal.smd_after[c]) for c in covs]

    order = np.argsort(before)
    covs_sorted = [covs[i] for i in order]
    before_sorted = [before[i] for i in order]
    after_sorted = [after[i] for i in order]

    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=150)
    y_pos = np.arange(len(covs_sorted))
    ax.scatter(before_sorted, y_pos, color="#DC2626", s=65, label="Unmatched (Before PSM)", zorder=3)
    ax.scatter(after_sorted, y_pos, color="#16A34A", s=65, marker="D", label="Matched (After PSM)", zorder=3)
    for i in range(len(covs_sorted)):
        ax.plot([after_sorted[i], before_sorted[i]], [y_pos[i], y_pos[i]], color="#CBD5E1", lw=1.2, zorder=2)
    ax.axvline(results.config.max_smd_threshold, color="#475569", ls="--", lw=1.3, label="|SMD| = 0.10 Threshold")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(covs_sorted)
    ax.set_xlabel("Absolute Standardized Mean Difference |SMD|")
    ax.set_title("Covariate Balance Love Plot (Before vs. After Matching)")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    return _save_or_return(fig, output_path)


def build_effect_comparison_figure(results: StudyResults, output_path: Path | None = None) -> Figure:
    ests = results.estimators
    names = [e.method for e in ests]
    points = [e.estimate for e in ests]
    err_low = [e.estimate - e.ci_low for e in ests]
    err_high = [e.ci_high - e.estimate for e in ests]

    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=150)
    y_pos = np.arange(len(names))[::-1]
    colors = ["#DC2626" if "Naive" in n else "#2563EB" for n in names]

    for y, pt, lo, hi, col in zip(y_pos, points, err_low, err_high, colors):
        ax.errorbar(pt, y, xerr=[[lo], [hi]], fmt="o", color=col, ecolor=col, elinewidth=2, capsize=4, markersize=7)

    ax.axvline(0.0, color="#6B7280", ls=":", lw=1.2, label="Zero Effect ($0)")
    ax.axvline(results.true_att_benchmark, color="#16A34A", ls="--", lw=1.4, label=f"True ATT Benchmark (${results.true_att_benchmark:,.0f})")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Estimated Earnings Effect (USD, 95% CI)")
    ax.set_title("Naive Selection Bias vs. Causal Treatment Effect Estimates")
    ax.legend(frameon=False, loc="lower left")
    ax.grid(axis="x", alpha=0.25)
    return _save_or_return(fig, output_path)


def render_markdown_report(results: StudyResults, locale: str = "en") -> str:
    """Build a plain-language causal report using Babel-localized strings and numbers."""
    sens = results.sensitivity
    bal = results.covariate_balance
    pm = results.propensity_metrics

    est_lines = []
    for est in results.estimators:
        label = t(f"metric_{est.method}", locale)
        ci_text = t("ci_range", locale).format(
            low=fmt_usd(est.ci_low, locale),
            high=fmt_usd(est.ci_high, locale),
        )
        est_lines.append(f"- **{label}:** {fmt_usd(est.estimate, locale)} ({ci_text})")

    sample_desc = t("units_treated_control", locale).format(
        total=results.n_total,
        treated=results.n_treated,
        control=results.n_control,
    )

    lines = [
        f"# {t('report_title', locale)}",
        "",
        f"## {t('exec_summary', locale)}",
        t("naive_bias_note", locale),
        "",
        t("causal_finding_note", locale),
        "",
        f"- **{t('sample_size_label', locale)}:** {sample_desc}",
        *est_lines,
        "",
        f"## {t('methodology_heading', locale)}",
        (
            f"- **{t('cv_auc_label', locale)}:** {fmt_num(pm.cv_auc_roc, locale)} "
            f"({t('precision_label', locale)}: {fmt_num(pm.cv_precision, locale)}, "
            f"{t('recall_label', locale)}: {fmt_num(pm.cv_recall, locale)})"
        ),
        (
            f"- **{t('overlap_label', locale)}:** {fmt_pct(pm.common_support_fraction, locale)} "
            f"({pm.treated_in_support} / {pm.control_in_support})"
        ),
        (
            f"- **{t('balance_before_after', locale)}:** "
            f"{fmt_num(bal.max_abs_smd_before, locale)} -> {fmt_num(bal.max_abs_smd_after, locale)}"
        ),
        f"- **{t('balance_status_label', locale)}:** {t('balance_ok' if bal.balanced else 'balance_fail', locale)}",
        "",
        f"## {t('sensitivity_heading', locale)}",
        f"- **{t('e_value_label', locale)}:** {fmt_num(sens.e_value_point, locale)} / {fmt_num(sens.e_value_ci, locale)}",
        f"- **{t('placebo_label', locale)}:** {fmt_usd(sens.placebo_effect, locale)} (p = {fmt_num(sens.placebo_p_value, locale, 4)})",
        f"- **{t('gamma_label', locale)}:** {fmt_num(sens.gamma_critical, locale, 2)}",
        "",
        f"## {t('limitations_heading', locale)}",
        t("limitations_body", locale),
        "",
    ]
    return "\n".join(lines)


def run_pipeline(
    config: StudyConfig | None = None,
    overwrite_data: bool = False,
    base_dir: Path = PROJECT_ROOT,
) -> Tuple[StudyResults, pd.DataFrame]:
    """Execute the full pipeline and persist dataset, figures, JSON results, and report."""
    cfg = config or StudyConfig()
    data_path = base_dir / "data" / "lalonde_study.csv"
    reports_dir = base_dir / "reports"
    figures_dir = reports_dir / "figures"

    df = load_or_generate_dataset(
        data_path=data_path,
        n_samples=cfg.n_samples,
        seed=cfg.seed,
        overwrite=overwrite_data,
        base_dir=base_dir,
    )
    results, ps, _ = run_full_study(df, config=cfg)

    fig_dir = validate_safe_path(figures_dir, base_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    build_dag_figure(fig_dir / "dag.png")
    build_propensity_overlap_figure(df, ps, fig_dir / "propensity_overlap.png")
    build_covariate_balance_figure(results, fig_dir / "covariate_balance.png")
    build_effect_comparison_figure(results, fig_dir / "effect_comparison.png")

    results_path = validate_safe_path(reports_dir / "results.json", base_dir)
    results_path.write_text(results.model_dump_json(indent=2), encoding="utf-8")

    report_path = validate_safe_path(reports_dir / "causal_report.md", base_dir)
    report_path.write_text(render_markdown_report(results, locale=cfg.locale), encoding="utf-8")
    return results, df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run causal inference observational study.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--locale", choices=["en", "id"], default="en")
    parser.add_argument("--caliper", type=float, default=0.2)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--overwrite-data", action="store_true")
    args = parser.parse_args()

    cfg = StudyConfig(
        seed=args.seed,
        locale=args.locale,
        caliper_sd_multiplier=args.caliper,
        n_bootstrap=args.bootstrap,
    )
    results, _ = run_pipeline(cfg, overwrite_data=args.overwrite_data)
    for est in results.estimators:
        print(f"{est.method}: ${est.estimate:,.2f} (95% CI: ${est.ci_low:,.2f} to ${est.ci_high:,.2f})")
    print(
        f"Propensity CV AUC={results.propensity_metrics.cv_auc_roc:.3f} | "
        f"Max |SMD| {results.covariate_balance.max_abs_smd_before:.3f} -> "
        f"{results.covariate_balance.max_abs_smd_after:.3f} | "
        f"Balanced={results.covariate_balance.balanced}"
    )


if __name__ == "__main__":
    main()
