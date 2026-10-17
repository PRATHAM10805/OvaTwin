"""Regenerate fallbackData.json with real patient data and complete metadata."""

import json
from datetime import datetime, timezone
from pathlib import Path
import math
import numpy as np
import pandas as pd

from src.integration.test_digital_twin import load_clinical_dataset
from src.Clinical.hpo_observation_builder import build_hpo_patient_values
from src.HPO.sbml_parameters import get_sbml_parameter_registry
from src.twin.digital_twin import PatientDigitalTwin

df = load_clinical_dataset()
selected_row = df.iloc[0]
patient = build_hpo_patient_values(selected_row)
twin = PatientDigitalTwin(patient)

init_status = twin.get_status()
baseline = twin.run_baseline(start_day=0, end_day=120, points=1201)
baseline_status = twin.get_status()

registry = get_sbml_parameter_registry()
parameter_id = registry.resolve_id("p1")
original_value = registry.get_baseline_value(parameter_id)
scenario_value = original_value * 1.001
scenario_name = "integration_test_p1_plus_0_1_percent"

scenario = twin.run_scenario(
    name=scenario_name,
    parameter_changes={parameter_id: scenario_value},
    start_day=0,
    end_day=120,
    points=1201,
)
scenario_status = twin.get_status()

comparison = twin.compare_baseline_scenario(scenario_name)
summary = twin.summary()

colnames = twin._column_names(baseline.result)
base_arr = np.asarray(baseline.result, dtype=float)
scen_arr = np.asarray(scenario.result, dtype=float)

baseline_shape = list(base_arr.shape)
baseline_finite = bool(np.isfinite(base_arr).all())
scenario_shape = list(scen_arr.shape)
scenario_finite = bool(np.isfinite(scen_arr).all())

step = 5
sample_indices = list(range(0, len(base_arr), step))
if sample_indices[-1] != len(base_arr) - 1:
    sample_indices.append(len(base_arr) - 1)

sampled_time = base_arr[sample_indices, colnames.index("time")].tolist()

trajectories = {}
for col in colnames:
    clean_col = col.strip("[]")
    idx = colnames.index(col)
    trajectories[clean_col] = {
        "baseline": [round(float(v), 5) for v in base_arr[sample_indices, idx]],
        "scenario": [round(float(v), 5) for v in scen_arr[sample_indices, idx]]
    }

all_metrics_finite = True
for var, m in comparison.items():
    for k in ["baseline_final", "scenario_final", "final_difference", "maximum_absolute_difference", "mean_absolute_difference", "rmse"]:
        val = m.get(k)
        if val is None or not math.isfinite(val):
            all_metrics_finite = False

assertions = [
    {"code": 'twin.get_status() == "initialized"', "passed": bool(init_status == "initialized"), "eval": str(init_status)},
    {"code": "baseline_values.shape == (1201, 46)", "passed": bool(baseline_shape == [1201, 46]), "eval": str(tuple(baseline_shape))},
    {"code": "np.isfinite(baseline_values).all()", "passed": bool(baseline_finite), "eval": str(baseline_finite)},
    {"code": 'twin.get_status() == "baseline_ready"', "passed": bool(baseline_status == "baseline_ready"), "eval": str(baseline_status)},
    {"code": "scenario_values.shape == baseline_values.shape", "passed": bool(scenario_shape == baseline_shape), "eval": f"{tuple(scenario_shape)} == {tuple(baseline_shape)}"},
    {"code": "np.isfinite(scenario_values).all()", "passed": bool(scenario_finite), "eval": str(scenario_finite)},
    {"code": 'twin.get_status() == "scenario_ready"', "passed": bool(scenario_status == "scenario_ready"), "eval": str(scenario_status)},
    {"code": "np.isfinite(metrics[...]) for all 45 species", "passed": bool(all_metrics_finite), "eval": f"{len(comparison)}/45 species finite"}
]

out_data = {
    "is_live": False,
    "is_fallback": True,
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "source_note": "Saved sample offline snapshot from Patient File No. 1. Not a live simulation run.",
    "summary": summary,
    "comparison_readiness": twin.profile.comparison_readiness(),
    "data_quality_flags": list(twin.profile.data_quality_flags),
    "patient_observations": {
        "patient_id": patient.patient_id,
        "row_index": 0,
        "age": patient.age,
        "weight": patient.weight,
        "height": patient.height,
        "bmi": patient.bmi,
        "waist_hip_ratio": patient.waist_hip_ratio,
        "lh": patient.lh,
        "fsh": patient.fsh,
        "fsh_lh_ratio": patient.fsh_lh_ratio,
        "amh": patient.amh,
        "progesterone": patient.progesterone,
        "prolactin": patient.prolactin,
        "tsh": patient.tsh,
        "cycle_status_encoded": patient.cycle_status_encoded,
        "cycle_length_encoded": patient.cycle_length_encoded,
        "total_follicles": patient.total_follicle_count,
        "left_follicles": patient.follicle_count_left,
        "right_follicles": patient.follicle_count_right,
        "mean_follicle_size": patient.mean_follicle_size,
        "endometrium": getattr(patient, "endometrium", None),
        "pcos_label": int(selected_row.get("PCOS (Y/N)")) if "PCOS (Y/N)" in selected_row.index and pd.notna(selected_row.get("PCOS (Y/N)")) else None,
    },
    "scenario_info": {
        "name": scenario_name,
        "parameter": parameter_id,
        "parameter_name": registry.get(parameter_id).name,
        "original_value": original_value,
        "scenario_value": scenario_value,
        "percent_change": 0.1
    },
    "comparison": {k.strip("[]"): v for k, v in comparison.items()},
    "all_species": [c.strip("[]") for c in colnames if c != "time"],
    "time": sampled_time,
    "trajectories": trajectories,
    "assertions": assertions
}

out_path = Path("frontend/src/fallbackData.json")
with open(out_path, "w") as f:
    json.dump(out_data, f, indent=2)

print("Saved updated fallbackData.json successfully!")
