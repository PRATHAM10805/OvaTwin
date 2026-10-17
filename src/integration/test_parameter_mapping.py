from pathlib import Path

import pandas as pd

from src.Clinical.hpo_observation_builder import (
    build_hpo_patient_values,
)
from src.twin.parameter_mapping import create_mapping


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


def test_patient_observations_map_to_correct_groups():
    df = load_clinical_dataset()
    patient = build_hpo_patient_values(df.iloc[0])

    mapping = create_mapping(patient)

    # The actual mapping interface uses these five observation groups.
    expected_groups = [
        "hormones",
        "follicular",
        "uterine",
        "cycle",
        "anthropometric",
    ]

    for group_name in expected_groups:
        assert hasattr(mapping, group_name), (
            f"Missing mapping group: {group_name}"
        )

        group = getattr(mapping, group_name)
        assert isinstance(group, dict)

    assert mapping.hormones["LH"] == patient.lh
    assert mapping.hormones["FSH"] == patient.fsh

    assert (
        mapping.hormones["FSH_LH_ratio"]
        == patient.fsh_lh_ratio
    )

    assert (
        mapping.follicular["total_count"]
        == patient.total_follicle_count
    )

    assert mapping.uterine["endometrium"] == patient.endometrium

    # Mapping organizes observations; it does not calibrate the SBML model.
    assert not hasattr(mapping, "parameters")