# Causal Inference Report: Job Training Program Impact on Earnings

## Executive Summary
A naive comparison of average earnings makes the training program appear harmful because participants started with significantly lower prior earnings and less schooling.

After controlling for baseline covariates via Propensity Score Matching, IPW, Doubly Robust (AIPW), and Difference-in-Differences, the estimated treatment effect is consistently positive.

- **Sample size:** 2675 units (298 treated, 2377 controls)
- **Naive Difference-in-Means:** -6,166.37 (95% CI: -6,806.81 to -5,525.93)
- **Propensity Score Matching (ATT):** 1,512.03 (95% CI: 1,088.21 to 1,915.24)
- **Inverse Probability Weighting (ATT):** 1,877.96 (95% CI: 1,078.73 to 2,595.50)
- **Doubly Robust AIPW (ATT):** 1,897.11 (95% CI: 1,589.73 to 2,204.50)
- **Difference-in-Differences (DiD):** 1,976.92 (95% CI: 1,782.22 to 2,171.63)

## Methodology & Assumptions
- **Propensity Model 5-Fold CV AUC-ROC:** 0.869 (Precision: 0.305, Recall: 0.792)
- **Common Support Overlap:** 75.8% (291 / 1737)
- **Max |SMD| Before -> After Matching:** 1.488 -> 0.093
- **Balance Status:** Post-matching covariate balance achieved (max |SMD| <= 0.10).

## Sensitivity & Robustness Analysis
- **E-Value (Point / Lower CI):** 1.801 / 1.621
- **Placebo Test (Controls Only):** -203.43 (p = 0.3775)
- **Critical Rosenbaum Gamma (p > 0.05):** 1.55

## Limitations
Causal estimates rely on conditional ignorability (no unmeasured confounders affecting both training enrollment and earnings growth), positivity (non-zero treatment probability across covariate strata), and SUTVA (no spillover across individuals). Difference-in-Differences additionally assumes parallel counterfactual earnings trajectories between 1975 and 1978.
