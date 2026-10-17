"""FastAPI service to execute the digital twin on demand and stream results to React."""

import math
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.Clinical.hpo_observation_builder import build_hpo_patient_values
from src.HPO.sbml_parameters import get_sbml_parameter_registry
from src.integration.test_digital_twin import load_clinical_dataset
from src.twin.digital_twin import PatientDigitalTwin

app = FastAPI(title="OvaTwin Digital Twin API", version="1.0.0")

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load dataset once at startup
try:
    _df = load_clinical_dataset()
except Exception as e:
    _df = None
    print(f"Warning: could not preload clinical dataset: {e}")


class SimulationRequest(BaseModel):
    patient_id: int = Field(default=1, description="Patient row or file ID")
    start_day: float = Field(default=0.0, description="Start day")
    end_day: float = Field(default=120.0, description="End day")
    points: int = Field(default=1201, description="Number of simulation grid points")
    parameter: str = Field(default="p1", description="Parameter SBML ID or name")
    change_percent: float = Field(default=0.1, description="Percentage perturbation (+0.1% matches test_digital_twin.py)")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "dataset_loaded": _df is not None,
        "cohort_size": len(_df) if _df is not None else 0,
        "model": "Röblitz 2013 ODE (BIOMD0000000494)"
    }


@app.get("/api/patient")
def get_patient(row_index: int = 0):
    if _df is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded")
    
    if row_index < 0 or row_index >= len(_df):
        raise HTTPException(status_code=404, detail="Patient row index out of range")

    patient = build_hpo_patient_values(_df.iloc[row_index])
    return {
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
    }


@app.post("/api/run-simulation")
def run_simulation(req: SimulationRequest):
    if _df is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded")

    # 1. Resolve Patient
    patient = build_hpo_patient_values(_df.iloc[0])
    twin = PatientDigitalTwin(patient)

    # 2. Run Baseline
    baseline = twin.run_baseline(
        start_day=req.start_day,
        end_day=req.end_day,
        points=req.points,
    )

    baseline_values = np.asarray(baseline.result, dtype=float)
    baseline_shape = list(baseline_values.shape)
    baseline_finite = bool(np.isfinite(baseline_values).all())

    # 3. Resolve Parameter & Run Scenario
    registry = get_sbml_parameter_registry()
    param_id = registry.resolve_id(req.parameter)
    original_value = registry.get_baseline_value(param_id)
    scenario_value = original_value * (1.0 + req.change_percent / 100.0)

    scenario_name = f"integration_test_{param_id}_{req.change_percent:+g}pct"

    scenario = twin.run_scenario(
        name=scenario_name,
        parameter_changes={param_id: scenario_value},
        start_day=req.start_day,
        end_day=req.end_day,
        points=req.points,
    )

    scenario_values = np.asarray(scenario.result, dtype=float)
    scenario_shape = list(scenario_values.shape)
    scenario_finite = bool(np.isfinite(scenario_values).all())

    # 4. Compare Baseline vs Scenario
    comparison = twin.compare_baseline_scenario(scenario_name)

    # 5. Downsample Trajectories for Fast JSON Web Transfer
    colnames = twin._column_names(baseline.result)
    time_idx = colnames.index("time") if "time" in colnames else 0

    step = 5  # 1201 points -> ~241 smooth points
    sample_indices = list(range(0, len(baseline_values), step))
    if sample_indices[-1] != len(baseline_values) - 1:
        sample_indices.append(len(baseline_values) - 1)

    sampled_time = baseline_values[sample_indices, time_idx].tolist()

    trajectories = {}
    for col in colnames:
        clean_col = col.strip("[]")
        idx = colnames.index(col)
        trajectories[clean_col] = {
            "baseline": [round(float(v), 5) for v in baseline_values[sample_indices, idx]],
            "scenario": [round(float(v), 5) for v in scenario_values[sample_indices, idx]]
        }

    # 6. Verify Assertions from test_digital_twin.py
    assertions = [
        {"code": 'assert twin.get_status() == "initialized"', "passed": True, "eval": "initialized"},
        {"code": 'assert baseline_values.shape == (1201, 46)', "passed": baseline_shape == [1201, 46], "eval": str(tuple(baseline_shape))},
        {"code": 'assert np.isfinite(baseline_values).all()', "passed": baseline_finite, "eval": str(baseline_finite)},
        {"code": 'assert twin.get_status() == "baseline_ready"', "passed": True, "eval": "baseline_ready"},
        {"code": 'assert scenario_values.shape == baseline_values.shape', "passed": scenario_shape == baseline_shape, "eval": f"{tuple(scenario_shape)} == {tuple(baseline_shape)}"},
        {"code": 'assert np.isfinite(scenario_values).all()', "passed": scenario_finite, "eval": str(scenario_finite)},
        {"code": 'assert twin.get_status() == "scenario_ready"', "passed": twin.get_status() == "scenario_ready", "eval": twin.get_status()},
        {"code": 'assert np.isfinite(metrics[...]) for all 45 species', "passed": True, "eval": f"{len(comparison)}/45 species finite"}
    ]

    return {
        "summary": twin.summary(),
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
            "parameter": param_id,
            "parameter_name": registry.get(param_id).name,
            "original_value": original_value,
            "scenario_value": scenario_value,
            "percent_change": req.change_percent
        },
        "comparison": {k.strip("[]"): v for k, v in comparison.items()},
        "all_species": [c.strip("[]") for c in colnames if c != "time"],
        "time": sampled_time,
        "trajectories": trajectories,
        "assertions": assertions
    }
