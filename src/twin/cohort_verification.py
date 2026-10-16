"""
cohort_verification.py

Phase 7.10: Full Cohort Batch Mapping & Robustness Verification.
Executes the validated 5-parameter personalization pipeline across all 541 patients
in the clinical dataset (PCOS_data_without_infertility-Full_new.csv).

Verifies:
  1. TC-04: Batch Pipeline Robustness (541/541 successful mappings, zero exceptions).
  2. Boundedness & Physiological Feasibility: 100% of parameters satisfy P > 0
     and 0.40 * P_base <= P <= 2.50 * P_base.
  3. Dynamic Stability: Representative cohort profiles (min, median, max, extreme PCOS)
     simulate cleanly for 120 days with zero solver crashes.
  4. Generation of cohort parameter registry and summary statistics artifacts.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from src.HPO.patient_values import HPOPatientValues
from src.Clinical.hpo_observation_builder import build_hpo_patient_values
from src.twin.parameter_mapping import create_mapping, PatientTwinMapping
from src.twin.digital_twin import PatientDigitalTwin
from src.HPO.sbml_parameters import get_sbml_parameter_registry
from src.HPO.parameters import get_authoritative_registry
from src.twin.simulation_guard import validate_trajectory_sanity

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = ROOT_DIR / "data" / "PCOS_data_without_infertility-Full_new.csv"

CORE_PARAMETER_IDS = ["p1", "p17", "p49", "p51", "p173"]


def run_cohort_mapping(
    csv_path: Optional[Path | str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes Core 5 parameter mapping across all 541 patients in the clinical dataset.
    Verifies that 100% of patients map successfully within physiological safety bounds.
    """
    path = Path(csv_path) if csv_path is not None else DEFAULT_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Clinical dataset not found at {path}")

    raw_df = pd.read_csv(path)
    total_patients = len(raw_df)

    auth_reg = get_authoritative_registry()
    sbml_reg = get_sbml_parameter_registry()

    patient_records: List[Dict[str, Any]] = []

    for idx, row in raw_df.iterrows():
        try:
            patient = build_hpo_patient_values(row)
            twin_map = create_mapping(patient, personalize_ode=True)
            ode_params = twin_map.ode_parameters

            record: Dict[str, Any] = {
                "patient_id": patient.patient_id,
                "fsh": patient.fsh,
                "lh": patient.lh,
                "fsh_lh_ratio": patient.fsh_lh_ratio,
                "total_follicles": patient.total_follicle_count,
                "mean_follicle_size": patient.mean_follicle_size,
                "cycle_regular": patient.cycle_regular,
                "amh": patient.amh,
            }

            for pid in CORE_PARAMETER_IDS:
                val = ode_params[pid]
                meta = auth_reg.get(pid)
                base_val = meta.baseline_value

                # Verify hard physiological bounds
                assert val > 0, f"Patient {patient.patient_id}: {pid} non-positive: {val}"
                assert meta.min_bound <= val <= meta.max_bound, (
                    f"Patient {patient.patient_id}: {pid}={val} outside bounds [{meta.min_bound}, {meta.max_bound}]"
                )
                assert not np.isnan(val) and not np.isinf(val), f"Patient {patient.patient_id}: {pid} is NaN/Inf"

                record[pid] = val
                record[f"{pid}_delta_pct"] = (val - base_val) / base_val * 100.0

            patient_records.append(record)

        except Exception as e:
            raise RuntimeError(f"Error mapping patient at index {idx} (ID {row.get('Patient File No.')}): {e}")

    cohort_df = pd.DataFrame(patient_records)

    # Compute summary statistics
    stats_list: List[Dict[str, Any]] = []
    for pid in CORE_PARAMETER_IDS:
        meta = auth_reg.get(pid)
        vals = cohort_df[pid]
        deltas = cohort_df[f"{pid}_delta_pct"]

        stats_list.append({
            "parameter_id": pid,
            "biological_name": meta.name,
            "baseline_value": meta.baseline_value,
            "hard_min": meta.min_bound,
            "hard_max": meta.max_bound,
            "cohort_mean": vals.mean(),
            "cohort_std": vals.std(),
            "cohort_min": vals.min(),
            "cohort_median": vals.median(),
            "cohort_max": vals.max(),
            "min_delta_pct": deltas.min(),
            "max_delta_pct": deltas.max(),
        })

    summary_df = pd.DataFrame(stats_list)

    return cohort_df, summary_df


def verify_representative_cohort_simulations(
    cohort_df: pd.DataFrame,
    csv_path: Optional[Path | str] = None,
) -> List[Dict[str, Any]]:
    """
    Selects diverse, representative patient phenotypes across the cohort
    (e.g. max LH dominance, low gonadotropin, extreme follicle count, follicular arrest)
    and verifies that each simulates cleanly for 120 days without solver crashes.
    """
    path = Path(csv_path) if csv_path is not None else DEFAULT_DATA_PATH
    raw_df = pd.read_csv(path)

    # Identify representative patient indices
    idx_p1_max = int(cohort_df["p1"].idxmax())
    idx_p1_min = int(cohort_df["p1"].idxmin())
    idx_p49_max = int(cohort_df["p49"].idxmax())
    idx_p51_min = int(cohort_df["p51"].idxmin())
    idx_patient_1 = 0

    representative_indices = {
        "Patient #1 (Cohort Baseline)": idx_patient_1,
        "Maximum LH Dominance (Classic PCOS)": idx_p1_max,
        "Minimum LH (Hypogonadotropic)": idx_p1_min,
        "Maximum Follicle Count (Polycystic Mass)": idx_p49_max,
        "Minimum Transition Rate (Follicular Arrest)": idx_p51_min,
    }

    verification_results = []

    for label, idx in representative_indices.items():
        row = raw_df.iloc[idx]
        patient = build_hpo_patient_values(row)
        twin = PatientDigitalTwin(patient)

        sim = twin.run_baseline(start_day=0, end_day=120, points=1201, personalized=True)
        is_success = sim.simulation_result.success if sim.simulation_result else (sim.result is not None)
        assert is_success is True, f"Simulation failed for {label}"

        sane, err, warnings = validate_trajectory_sanity(sim.result, expected_points=1201)
        assert sane is True, f"Trajectory sanity failed for {label}: {err}"

        verification_results.append({
            "profile_label": label,
            "patient_id": patient.patient_id,
            "simulation_success": is_success,
            "trajectory_shape": sim.result.shape,
            "sanity_passed": sane,
            "warnings_count": len(warnings),
            "p1": twin.mapping.ode_parameters["p1"],
            "p17": twin.mapping.ode_parameters["p17"],
            "p49": twin.mapping.ode_parameters["p49"],
            "p51": twin.mapping.ode_parameters["p51"],
            "p173": twin.mapping.ode_parameters["p173"],
        })

    return verification_results


def export_cohort_artifacts(
    cohort_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Optional[Path | str] = None,
) -> Tuple[Path, Path]:
    """Exports cohort parameter values and statistical summary to CSV."""
    out_dir = Path(output_dir) if output_dir is not None else (ROOT_DIR / "results")
    out_dir.mkdir(parents=True, exist_ok=True)

    cohort_csv = out_dir / "cohort_core_5_parameters.csv"
    summary_csv = out_dir / "cohort_core_5_summary_stats.csv"

    cohort_df.to_csv(cohort_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    return cohort_csv, summary_csv
