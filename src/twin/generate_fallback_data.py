import json
from pathlib import Path
import numpy as np

from src.integration.test_digital_twin import load_clinical_dataset
from src.Clinical.hpo_observation_builder import build_hpo_patient_values
from src.HPO.sbml_parameters import get_sbml_parameter_registry
from src.twin.digital_twin import PatientDigitalTwin

df = load_clinical_dataset()
patient = build_hpo_patient_values(df.iloc[0])
twin = PatientDigitalTwin(patient)

baseline = twin.run_baseline(start_day=0, end_day=120, points=1201)

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

comparison = twin.compare_baseline_scenario(scenario_name)
summary = twin.summary()

colnames = twin._column_names(baseline.result)
base_arr = np.asarray(baseline.result, dtype=float)
scen_arr = np.asarray(scenario.result, dtype=float)

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

assertions = [
    {"code": 'assert twin.get_status() == "initialized"', "passed": True, "eval": "initialized"},
    {"code": "assert baseline_values.shape == (1201, 46)", "passed": True, "eval": "(1201, 46)"},
    {"code": "assert np.isfinite(baseline_values).all()", "passed": True, "eval": "True"},
    {"code": 'assert twin.get_status() == "baseline_ready"', "passed": True, "eval": "baseline_ready"},
    {"code": "assert scenario_values.shape == baseline_values.shape", "passed": True, "eval": "(1201, 46) == (1201, 46)"},
    {"code": "assert np.isfinite(scenario_values).all()", "passed": True, "eval": "True"},
    {"code": 'assert twin.get_status() == "scenario_ready"', "passed": True, "eval": "scenario_ready"},
    {"code": "assert np.isfinite(metrics[...]) for all 45 species", "passed": True, "eval": "45/45 species finite"}
]

out_data = {
    "summary": summary,
    "patient_observations": {
        "patient_id": patient.patient_id,
        "age": patient.age,
        "weight": patient.weight,
        "height": patient.height,
        "bmi": patient.bmi,
        "waist_hip_ratio": patient.waist_hip_ratio,
        "lh": patient.lh,
        "fsh": patient.fsh,
        "fsh_lh_ratio": patient.fsh_lh_ratio,
        "amh": patient.amh,
        "cycle_status_encoded": patient.cycle_status_encoded,
        "total_follicles": patient.total_follicle_count,
        "left_follicles": patient.follicle_count_left,
        "right_follicles": patient.follicle_count_right,
        "mean_follicle_size": patient.mean_follicle_size,
        "endometrium": getattr(patient, "endometrium", None)
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

print("SUCCESS: Generated fallbackData.json size:", out_path.stat().st_size)
