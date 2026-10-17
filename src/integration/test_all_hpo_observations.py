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


def test_all_patients_can_be_converted_and_validated():
    df = load_clinical_dataset()

    assert len(df) == 541

    patient_ids = []
    valid_count = 0
    invalid_count = 0
    missing_summary = {}

    for _, row in df.iterrows():
        patient = build_hpo_patient_values(row)

        assert isinstance(patient.patient_id, int)
        assert patient.patient_id > 0

        patient_ids.append(patient.patient_id)

        result = validate_patient_values(patient)

        assert isinstance(result, dict)
        assert "valid" in result
        assert "missing" in result

        if result["valid"]:
            valid_count += 1
        else:
            invalid_count += 1

            for field in result["missing"]:
                missing_summary[field] = (
                    missing_summary.get(field, 0) + 1
                )

    assert valid_count + invalid_count == len(df)
    assert len(set(patient_ids)) == len(df)

    print("\nHPO OBSERVATION VALIDATION")
    print("=" * 60)
    print(f"Total patients: {len(df)}")
    print(f"Valid patients: {valid_count}")
    print(f"Invalid patients: {invalid_count}")
    print(f"Unique patient IDs: {len(set(patient_ids))}")

    print("\nMissing required observations:")
    print(missing_summary or "None")