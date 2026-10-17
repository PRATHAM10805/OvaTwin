from pathlib import Path

import pandas as pd

from src.Clinical.hpo_observation_builder import (
    build_hpo_patient_values,
)
from src.HPO.validation import validate_patient_values


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREPROCESSED_PATH = (
    PROJECT_ROOT / "data" / "clinical_dataset_preprocessed.csv"
)
RAW_PATH = (
    PROJECT_ROOT
    / "data"
    / "PCOS_data_without_infertility-Full_new.csv"
)


def load_clinical_dataset():
    """Prefer derived features from preprocessing."""
    path = (
        PREPROCESSED_PATH
        if PREPROCESSED_PATH.is_file()
        else RAW_PATH
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Clinical dataset not found: {path}"
        )

    return pd.read_csv(path)


def test_patient_observations_are_built_correctly():
    df = load_clinical_dataset()

    assert not df.empty
    assert "Patient File No." in df.columns

    patient = build_hpo_patient_values(df.iloc[0])

    assert patient.patient_id == 1
    assert patient.age is not None
    assert patient.weight is not None
    assert patient.height is not None
    assert patient.bmi is not None
    assert patient.fsh is not None
    assert patient.lh is not None
    assert patient.fsh_lh_ratio is not None

    assert patient.fsh_lh_ratio >= 0

    # Cycle codes must remain encoded until their meaning is verified.
    assert patient.cycle_regular is None
    assert patient.cycle_status_encoded is not None


def test_patient_validation_returns_expected_structure():
    df = load_clinical_dataset()
    patient = build_hpo_patient_values(df.iloc[0])

    result = validate_patient_values(patient)

    assert isinstance(result, dict)
    assert "valid" in result
    assert "missing" in result
    assert isinstance(result["valid"], bool)
    assert isinstance(result["missing"], (list, tuple))