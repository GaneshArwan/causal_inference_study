"""Causal inference study package for observational job-training evaluation."""

from src.causal_engine import run_full_study
from src.data_generator import generate_lalonde_dataset, load_or_generate_dataset
from src.schemas import StudyConfig, StudyResults

__all__ = [
    "StudyConfig",
    "StudyResults",
    "generate_lalonde_dataset",
    "load_or_generate_dataset",
    "run_full_study",
]
