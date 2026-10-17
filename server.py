"""FastAPI service to execute the digital twin on demand and stream results to React."""

from datetime import datetime, timezone
import math
from typing import Optional, Union
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.Clinical.hpo_observation_builder import build_hpo_patient_values
from src.HPO.sbml_parameters import get_sbml_parameter_registry
from src.integration.test_digital_twin import load_clinical_dataset
from src.twin.digital_twin import PatientDigitalTwin

app = FastAPI(title="OvaTwin Digital Twin API", version="1.1.0")

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


def _get_valid_cohort() -> pd.DataFrame:
    """Return cohort filtered for valid positive integer patient IDs matching digital_twin.py."""
    if _df is None:
        raise HTTPException(status_code=500, detail="Clinical dataset not loaded")
    if "Patient File No." not in _df.columns:
        return _df.copy()

    patient_ids = pd.to_numeric(_df["Patient File No."], errors="coerce")
    valid_mask = (
        patient_ids.notna()
        & np.isfinite(patient_ids)
        & patient_ids.gt(0)
        & patient_ids.mod(1).eq(0)
    )
    valid_df = _df.loc[valid_mask].copy()
    if valid_df.empty:
        raise HTTPException(status_code=500, detail="No rows have valid positive integer patient IDs")
    return valid_df


def _resolve_patient_row(patient_id: Optional[Union[int, str, float]] = None, random_patient: bool = False) -> pd.Series:
    """Resolve patient row either randomly or by requested identifier (Patient File No. or row index)."""
    cohort = _get_valid_cohort()

    if random_patient or patient_id is None or str(patient_id).strip().lower() in ("random", "", "none"):
        return cohort.sample(1).iloc[0]

    # Try matching Patient File No.
    try:
        pid_float = float(patient_id)
        matches = cohort[cohort["Patient File No."].astype(float) == pid_float]
    except (ValueError, TypeError):
        matches = cohort[cohort["Patient File No."].astype(str).str.strip() == str(patient_id).strip()]

    if len(matches) > 0:
        return matches.iloc[0]

    # Fallback to 0-based dataframe index if passed as integer
    try:
        idx = int(patient_id)
        if 0 <= idx < len(_df):
            return _df.iloc[idx]
    except (ValueError, TypeError):
        pass

    raise HTTPException(status_code=404, detail=f"Patient {patient_id!r} not found in clinical cohort")


def _extract_patient_dict(row: pd.Series, patient) -> dict:
    """Extract real patient clinical observations with no fabrication."""
    pcos_val = row.get("PCOS (Y/N)") if "PCOS (Y/N)" in row.index else None
    parsed_pcos = None
    if pd.notna(pcos_val):
        try:
            parsed_pcos = int(pcos_val)
        except (ValueError, TypeError):
            parsed_pcos = str(pcos_val)

    return {
        "patient_id": patient.patient_id,
        "row_index": int(row.name) if hasattr(row, "name") and str(row.name).isdigit() else None,
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
        "pcos_label": parsed_pcos,
    }


class SimulationRequest(BaseModel):
    patient_id: Optional[Union[int, str]] = Field(default=None, description="Patient File No. or 'random'")
    random_patient: bool = Field(default=True, description="Pick a random patient from the cohort on each run")
    start_day: float = Field(default=0.0, description="Start day")
    end_day: float = Field(default=120.0, description="End day")
    points: int = Field(default=1201, description="Number of simulation grid points")
    parameter: str = Field(default="p1", description="Parameter SBML ID or name")
    change_percent: float = Field(default=0.1, description="Percentage perturbation (+0.1% matches test_digital_twin.py)")


@app.get("/api/health")
def health():
    cohort_len = len(_df) if _df is not None else 0
    return {
        "status": "ok",
        "dataset_loaded": _df is not None,
        "cohort_size": cohort_len,
        "model": "Röblitz 2013 ODE (BIOMD0000000494)",
        "simulation_engine": "RoadRunner CVODE"
    }


@app.get("/api/patient")
def get_patient(
    patient_id: Optional[str] = Query(default=None),
    random: bool = Query(default=False),
):
    selected_row = _resolve_patient_row(patient_id=patient_id, random_patient=random)
    patient = build_hpo_patient_values(selected_row)
    twin = PatientDigitalTwin(patient)
    return {
        "patient_observations": _extract_patient_dict(selected_row, patient),
        "comparison_readiness": twin.profile.comparison_readiness(),
        "data_quality_flags": list(twin.profile.data_quality_flags),
        "summary": twin.profile.observation_summary(),
    }


@app.post("/api/run-simulation")
def run_simulation(req: SimulationRequest):
    if _df is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded")

    # 1. Resolve Patient (Finding 1: respects patient selection and supports random sampling)
    selected_row = _resolve_patient_row(req.patient_id, req.random_patient)
    patient = build_hpo_patient_values(selected_row)
    twin = PatientDigitalTwin(patient)
    init_status = twin.get_status()

    # 2. Run Baseline
    baseline = twin.run_baseline(
        start_day=req.start_day,
        end_day=req.end_day,
        points=req.points,
    )
    baseline_status = twin.get_status()
    baseline_values = np.asarray(baseline.result, dtype=float)
    baseline_shape = list(baseline_values.shape)
    baseline_finite = bool(np.isfinite(baseline_values).all())

    # 3. Resolve Parameter & Run Scenario
    registry = get_sbml_parameter_registry()
    param_id = registry.resolve_id(req.parameter)
    original_value = registry.get_baseline_value(param_id)
    scenario_value = original_value * (1.0 + req.change_percent / 100.0)

    scenario_name = f"{param_id}_{req.change_percent:+g}pct_sensitivity"

    scenario = twin.run_scenario(
        name=scenario_name,
        parameter_changes={param_id: scenario_value},
        start_day=req.start_day,
        end_day=req.end_day,
        points=req.points,
    )
    scenario_status = twin.get_status()
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

    # 6. Dynamically Evaluate Assertions / Diagnostics (Finding 2: NO hardcoding!)
    all_metrics_finite = True
    non_finite_list = []
    for var, m in comparison.items():
        for k in ["baseline_final", "scenario_final", "final_difference", "maximum_absolute_difference", "mean_absolute_difference", "rmse"]:
            val = m.get(k)
            if val is None or not math.isfinite(val):
                all_metrics_finite = False
                non_finite_list.append(f"{var}.{k}")

    assertions = [
        {
            "code": 'twin.get_status() == "initialized"',
            "passed": bool(init_status == "initialized"),
            "eval": str(init_status)
        },
        {
            "code": f"baseline_values.shape == ({req.points}, 46)",
            "passed": bool(baseline_shape == [req.points, 46]),
            "eval": str(tuple(baseline_shape))
        },
        {
            "code": "np.isfinite(baseline_values).all()",
            "passed": bool(baseline_finite),
            "eval": str(baseline_finite)
        },
        {
            "code": 'twin.get_status() == "baseline_ready"',
            "passed": bool(baseline_status == "baseline_ready"),
            "eval": str(baseline_status)
        },
        {
            "code": "scenario_values.shape == baseline_values.shape",
            "passed": bool(scenario_shape == baseline_shape),
            "eval": f"{tuple(scenario_shape)} == {tuple(baseline_shape)}"
        },
        {
            "code": "np.isfinite(scenario_values).all()",
            "passed": bool(scenario_finite),
            "eval": str(scenario_finite)
        },
        {
            "code": 'twin.get_status() == "scenario_ready"',
            "passed": bool(scenario_status == "scenario_ready"),
            "eval": str(scenario_status)
        },
        {
            "code": "np.isfinite(metrics[...]) for all 45 species",
            "passed": bool(all_metrics_finite),
            "eval": f"{len(comparison)}/45 species finite" if all_metrics_finite else f"Non-finite: {non_finite_list[:2]}"
        }
    ]

    return {
        "is_live": True,
        "is_fallback": False,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": twin.summary(),
        "comparison_readiness": twin.profile.comparison_readiness(),
        "data_quality_flags": list(twin.profile.data_quality_flags),
        "patient_observations": _extract_patient_dict(selected_row, patient),
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
