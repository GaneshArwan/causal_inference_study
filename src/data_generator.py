"""Deterministic LaLonde-calibrated observational dataset generator."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.guardrails import validate_dataframe, validate_safe_path

TRUE_ATT_BENCHMARK = 1800.0
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "lalonde_study.csv"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def generate_lalonde_dataset(
    n_samples: int = 2675,
    seed: int = 42,
    true_att: float = TRUE_ATT_BENCHMARK,
) -> pd.DataFrame:
    """Generate a deterministic LaLonde-style observational dataset with known ATT.

    Calibrated to the Dehejia-Wahba NSW + PSID observational structure (~7% treated),
    where lower baseline earnings and education increase treatment enrollment
    (creating strong negative selection bias in naive comparisons) while satisfying
    conditional ignorability and parallel trends.
    """
    if n_samples < 200:
        raise ValueError("n_samples must be at least 200.")

    rng = np.random.default_rng(seed)

    # Latent baseline socioeconomic factor (mean 0, sd 1)
    ses = rng.normal(0.0, 1.0, size=n_samples)

    age = np.clip(np.round(34.0 + 5.0 * ses + rng.normal(0.0, 7.0, n_samples)), 18, 55).astype(int)
    education = np.clip(np.round(11.8 + 1.4 * ses + rng.normal(0.0, 1.7, n_samples)), 6, 18).astype(int)
    nodegree = (education < 12).astype(int)

    married_prob = _sigmoid(0.35 + 0.85 * ses)
    married = rng.binomial(1, married_prob).astype(int)

    black_prob = _sigmoid(-1.35 - 0.75 * ses)
    black = rng.binomial(1, black_prob).astype(int)

    hispanic_prob = _sigmoid(-1.95 - 0.30 * ses)
    hispanic = np.where(black == 0, rng.binomial(1, hispanic_prob), 0).astype(int)

    re74 = np.maximum(
        0.0,
        13500.0
        + 5200.0 * ses
        + 1100.0 * married
        - 1600.0 * nodegree
        - 900.0 * black
        + rng.normal(0.0, 2400.0, n_samples),
    )
    re75 = np.maximum(0.0, re74 + 350.0 + rng.normal(0.0, 1400.0, n_samples))

    # Treatment assignment strictly a function of observed covariates (ignorability holds)
    z_re75 = (re75 - re75.mean()) / (re75.std() + 1e-8)
    z_edu = (education - education.mean()) / (education.std() + 1e-8)
    z_age = (age - age.mean()) / (age.std() + 1e-8)

    logit_treat = (
        -3.15
        - 1.05 * z_re75
        - 0.45 * z_edu
        - 0.25 * z_age
        + 0.45 * nodegree
        - 0.40 * married
        + 0.55 * black
    )
    treat = rng.binomial(1, _sigmoid(logit_treat)).astype(int)

    # Control potential outcome preserves parallel trends on average plus realistic noise
    re78_control = np.maximum(0.0, re75 + 550.0 + rng.normal(0.0, 1650.0, n_samples))
    individual_te = rng.normal(true_att, 350.0, n_samples)
    re78 = np.maximum(0.0, re78_control + treat * individual_te)

    df = pd.DataFrame(
        {
            "treat": treat,
            "age": age,
            "education": education,
            "black": black,
            "hispanic": hispanic,
            "married": married,
            "nodegree": nodegree,
            "re74": np.round(re74, 2),
            "re75": np.round(re75, 2),
            "re78": np.round(re78, 2),
        }
    )
    return validate_dataframe(df)


def load_or_generate_dataset(
    data_path: Path = DEFAULT_DATA_PATH,
    n_samples: int = 2675,
    seed: int = 42,
    overwrite: bool = False,
    base_dir: Path = PROJECT_ROOT,
) -> pd.DataFrame:
    """Load validated CSV from disk or generate and persist deterministically."""
    safe_path = validate_safe_path(data_path, base_dir)
    if safe_path.exists() and not overwrite:
        df = pd.read_csv(safe_path)
        return validate_dataframe(df)

    safe_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_lalonde_dataset(n_samples=n_samples, seed=seed)
    df.to_csv(safe_path, index=False)
    return df
