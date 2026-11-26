"""Internationalization helpers backed by Babel for English and Indonesian."""

from typing import Dict

from babel.numbers import format_currency, format_decimal, format_percent

SUPPORTED_LOCALES = ("en", "id")

_BABEL_LOCALE_MAP: Dict[str, str] = {
    "en": "en_US",
    "id": "id_ID",
}

MESSAGES: Dict[str, Dict[str, str]] = {
    "en": {
        "app_title": "Causal Inference Study: Job Training Earnings Impact",
        "app_subtitle": "Separating true causal treatment effects from selection bias in observational data.",
        "report_title": "Causal Inference Report: Job Training Program Impact on Earnings",
        "exec_summary": "Executive Summary",
        "naive_bias_note": (
            "A naive comparison of average earnings makes the training program appear harmful "
            "because participants started with significantly lower prior earnings and less schooling."
        ),
        "causal_finding_note": (
            "After controlling for baseline covariates via Propensity Score Matching, IPW, "
            "Doubly Robust (AIPW), and Difference-in-Differences, the estimated treatment effect "
            "is consistently positive."
        ),
        "methodology_heading": "Methodology & Assumptions",
        "sensitivity_heading": "Sensitivity & Robustness Analysis",
        "limitations_heading": "Limitations",
        "limitations_body": (
            "Causal estimates rely on conditional ignorability (no unmeasured confounders affecting "
            "both training enrollment and earnings growth), positivity (non-zero treatment probability "
            "across covariate strata), and SUTVA (no spillover across individuals). "
            "Difference-in-Differences additionally assumes parallel counterfactual earnings trajectories "
            "between 1975 and 1978."
        ),
        "sample_size_label": "Sample size",
        "units_treated_control": "{total} units ({treated} treated, {control} controls)",
        "ci_range": "95% CI: {low} to {high}",
        "cv_auc_label": "Propensity Model 5-Fold CV AUC-ROC",
        "precision_label": "Precision",
        "recall_label": "Recall",
        "overlap_label": "Common Support Overlap",
        "balance_before_after": "Max |SMD| Before -> After Matching",
        "balance_status_label": "Balance Status",
        "e_value_label": "E-Value (Point / Lower CI)",
        "placebo_label": "Placebo Test (Controls Only)",
        "gamma_label": "Critical Rosenbaum Gamma (p > 0.05)",
        "metric_Naive Difference-in-Means": "Naive Difference-in-Means",
        "metric_Propensity Score Matching (ATT)": "Propensity Score Matching (ATT)",
        "metric_Inverse Probability Weighting (ATT)": "Inverse Probability Weighting (ATT)",
        "metric_Doubly Robust AIPW (ATT)": "Doubly Robust AIPW (ATT)",
        "metric_Difference-in-Differences (DiD)": "Difference-in-Differences (DiD)",
        "caliper_label": "Matching Caliper (SD of logit PS)",
        "bootstrap_label": "Bootstrap Replicates",
        "locale_label": "Language / Bahasa",
        "fig_balance_caption": "Covariate Balance (Love Plot)",
        "fig_effect_caption": "Naive vs. Causal Estimators",
        "balance_ok": "Post-matching covariate balance achieved (max |SMD| <= 0.10).",
        "balance_fail": "Covariate imbalance remains above 0.10 threshold.",
    },
    "id": {
        "app_title": "Studi Inferensi Kausal: Dampak Pelatihan Kerja terhadap Pendapatan",
        "app_subtitle": "Memisahkan efek kausal sebenarnya dari bias seleksi pada data observasional.",
        "report_title": "Laporan Inferensi Kausal: Dampak Program Pelatihan Kerja pada Pendapatan",
        "exec_summary": "Ringkasan Eksekutif",
        "naive_bias_note": (
            "Perbandingan rata-rata secara naif membuat program pelatihan tampak merugikan "
            "karena peserta memiliki pendapatan awal dan tingkat pendidikan yang jauh lebih rendah."
        ),
        "causal_finding_note": (
            "Setelah mengontrol kovariat dasar melalui Propensity Score Matching, IPW, "
            "Doubly Robust (AIPW), dan Difference-in-Differences, estimasi efek perlakuan "
            "konsisten bernilai positif."
        ),
        "methodology_heading": "Metodologi & Asumsi",
        "sensitivity_heading": "Analisis Sensitivitas & Ketahanan",
        "limitations_heading": "Keterbatasan",
        "limitations_body": (
            "Estimasi kausal bergantung pada asumsi conditional ignorability (tidak ada pembaur "
            "tak terukur yang memengaruhi partisipasi pelatihan dan pertumbuhan pendapatan), "
            "positivity (peluang perlakuan bukan nol di setiap strata kovariat), serta SUTVA "
            "(tidak ada efek limpahan antar individu). Difference-in-Differences juga mengasumsikan "
            "tren pendapatan kontrafaktual yang sejajar antara 1975 dan 1978."
        ),
        "sample_size_label": "Ukuran sampel",
        "units_treated_control": "{total} unit ({treated} perlakuan, {control} kontrol)",
        "ci_range": "IK 95%: {low} hingga {high}",
        "cv_auc_label": "AUC-ROC CV 5-Fold Model Propensity",
        "precision_label": "Presisi",
        "recall_label": "Recall",
        "overlap_label": "Irisan Common Support",
        "balance_before_after": "Maks |SMD| Sebelum -> Sesudah Pencocokan",
        "balance_status_label": "Status Keseimbangan",
        "e_value_label": "E-Value (Titik / Batas Bawah IK)",
        "placebo_label": "Uji Plasebo (Khusus Kontrol)",
        "gamma_label": "Gamma Rosenbaum Kritis (p > 0.05)",
        "metric_Naive Difference-in-Means": "Selisih Rata-Rata Naif",
        "metric_Propensity Score Matching (ATT)": "Propensity Score Matching (ATT)",
        "metric_Inverse Probability Weighting (ATT)": "Inverse Probability Weighting (ATT)",
        "metric_Doubly Robust AIPW (ATT)": "Doubly Robust AIPW (ATT)",
        "metric_Difference-in-Differences (DiD)": "Difference-in-Differences (DiD)",
        "caliper_label": "Caliper Pencocokan (SD logit PS)",
        "bootstrap_label": "Ulangan Bootstrap",
        "locale_label": "Bahasa / Language",
        "fig_balance_caption": "Keseimbangan Kovariat (Love Plot)",
        "fig_effect_caption": "Estimator Naif vs. Kausal",
        "balance_ok": "Keseimbangan kovariat pasca-pencocokan tercapai (maks |SMD| <= 0.10).",
        "balance_fail": "Ketidakseimbangan kovariat masih melebihi ambang 0.10.",
    },
}


def t(key: str, locale: str = "en") -> str:
    """Translate a message key with safe fallback to English."""
    lang = locale if locale in SUPPORTED_LOCALES else "en"
    return MESSAGES[lang].get(key, MESSAGES["en"].get(key, key))


def fmt_usd(amount: float, locale: str = "en") -> str:
    """Format a dollar value using Babel locale conventions."""
    babel_loc = _BABEL_LOCALE_MAP.get(locale, "en_US")
    return format_currency(amount, "USD", locale=babel_loc, format="#,##0")


def fmt_num(value: float, locale: str = "en", decimals: int = 3) -> str:
    """Format a decimal number using Babel locale rules."""
    babel_loc = _BABEL_LOCALE_MAP.get(locale, "en_US")
    pattern = "#,##0." + ("0" * decimals) if decimals > 0 else "#,##0"
    return format_decimal(round(value, decimals), format=pattern, locale=babel_loc)


def fmt_pct(value: float, locale: str = "en") -> str:
    """Format a proportion [0, 1] as a percentage string via Babel."""
    babel_loc = _BABEL_LOCALE_MAP.get(locale, "en_US")
    return format_percent(value, format="0.0%", locale=babel_loc)
