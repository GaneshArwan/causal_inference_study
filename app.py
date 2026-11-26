"""Interactive Streamlit dashboard for the Causal Inference Study."""

import matplotlib.pyplot as plt
import streamlit as st

from src.causal_engine import run_full_study
from src.data_generator import DEFAULT_DATA_PATH, load_or_generate_dataset
from src.i18n import fmt_num, fmt_pct, fmt_usd, t
from src.schemas import StudyConfig
from src.train import (
    build_covariate_balance_figure,
    build_effect_comparison_figure,
    render_markdown_report,
)

st.set_page_config(page_title="Causal Inference Study", layout="wide")


def main() -> None:
    with st.sidebar:
        locale = st.selectbox(t("locale_label", "en"), options=["en", "id"], index=0)
        caliper = st.slider(t("caliper_label", locale), min_value=0.05, max_value=1.0, value=0.2, step=0.05)
        n_boot = st.slider(t("bootstrap_label", locale), min_value=50, max_value=500, value=150, step=50)

    cfg = StudyConfig(caliper_sd_multiplier=caliper, n_bootstrap=n_boot, locale=locale)
    df = load_or_generate_dataset(DEFAULT_DATA_PATH, n_samples=cfg.n_samples, seed=cfg.seed)
    results, _, _ = run_full_study(df, config=cfg)

    st.title(t("app_title", locale))
    st.caption(t("app_subtitle", locale))

    cols = st.columns(5)
    for col, est in zip(cols, results.estimators):
        col.metric(label=t(f"metric_{est.method}", locale), value=fmt_usd(est.estimate, locale))

    bal = results.covariate_balance
    pm = results.propensity_metrics
    st.info(
        f"{t('cv_auc_label', locale)}: {fmt_num(pm.cv_auc_roc, locale)} | "
        f"{t('overlap_label', locale)}: {fmt_pct(pm.common_support_fraction, locale)} | "
        f"{t('balance_before_after', locale)}: {fmt_num(bal.max_abs_smd_before, locale)} -> "
        f"{fmt_num(bal.max_abs_smd_after, locale)} | "
        f"{t('balance_ok' if bal.balanced else 'balance_fail', locale)}"
    )

    fig_cols = st.columns(2)
    fig_bal = build_covariate_balance_figure(results)
    fig_eff = build_effect_comparison_figure(results)
    fig_cols[0].pyplot(fig_bal)
    fig_cols[0].caption(t("fig_balance_caption", locale))
    fig_cols[1].pyplot(fig_eff)
    fig_cols[1].caption(t("fig_effect_caption", locale))
    plt.close(fig_bal)
    plt.close(fig_eff)

    st.markdown(render_markdown_report(results, locale=locale))


if __name__ == "__main__":
    main()
