from pathlib import Path

import numpy as np
import pandas as pd

from src.Clinical.hpo_observation_builder import (
    build_hpo_patient_values,
)
from src.HPO.sbml_parameters import (
    get_sbml_parameter_registry,
)
from src.twin.digital_twin import PatientDigitalTwin


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


def test_digital_twin_baseline_and_scenario():
    df = load_clinical_dataset()

    patient = build_hpo_patient_values(df.iloc[0])
    twin = PatientDigitalTwin(patient)

    assert twin.get_status() == "initialized"
    assert twin.profile.patient.patient_id == patient.patient_id

    baseline = twin.run_baseline(
        start_day=0,
        end_day=120,
        points=1201,
    )

    baseline_values = np.asarray(
        baseline.result,
        dtype=float,
    )

    assert baseline_values.shape == (1201, 46)
    assert np.isfinite(baseline_values).all()
    assert twin.get_status() == "baseline_ready"

    # Use the verified parameter registry rather than assuming
    # a parameter's baseline value.
    registry = get_sbml_parameter_registry()

    parameter_id = registry.resolve_id("p1")
    original_value = registry.get_baseline_value(parameter_id)

    scenario_value = original_value * 1.001

    scenario_name = "integration_test_p1_plus_0_1_percent"

    scenario = twin.run_scenario(
        name=scenario_name,
        parameter_changes={
            parameter_id: scenario_value,
        },
        start_day=0,
        end_day=120,
        points=1201,
    )

    scenario_values = np.asarray(
        scenario.result,
        dtype=float,
    )

    assert scenario_values.shape == baseline_values.shape
    assert np.isfinite(scenario_values).all()
    assert twin.get_status() == "scenario_ready"

    comparison = twin.compare_baseline_scenario(scenario_name)

    assert comparison

    for variable, metrics in comparison.items():
        assert np.isfinite(metrics["baseline_final"])
        assert np.isfinite(metrics["scenario_final"])
        assert np.isfinite(metrics["final_difference"])
        assert np.isfinite(metrics["maximum_absolute_difference"])
        assert np.isfinite(metrics["mean_absolute_difference"])
        assert np.isfinite(metrics["rmse"])

    assert twin.summary()["baseline_available"] is True
    assert scenario_name in twin.summary()["scenario_names"]