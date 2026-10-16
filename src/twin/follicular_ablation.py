"""
follicular_ablation.py

Phase 7.9: Follicular & Combined Core 5 Ablation Simulation Engine.
Supports:
  1. Isolated single-parameter ablation for follicular parameters:
     - p49 (k_AF1)       <- Antral follicle count (TC-A3)
     - p51 (k_AF2_AF3)   <- Follicle size / cycle regularity (TC-A4)
     - p173 (b_IhB)      <- Follicle count / AMH (TC-A5)
  2. Full combined Core 5 personalized baseline simulation (TC-A6):
     p1, p17, p49, p51, p173 applied simultaneously.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd

from src.HPO.patient_values import HPOPatientValues
from src.twin.digital_twin import PatientDigitalTwin
from src.HPO.sbml_parameters import get_sbml_parameter_registry
from src.twin.simulation_guard import validate_trajectory_sanity

ROOT_DIR = Path(__file__).resolve().parents[2]

KEY_PHYSIOLOGICAL_SPECIES = [
    "LH_bld",     # Circulating blood LH
    "LH_Pit",     # Pituitary LH reservoir
    "FSH_bld",    # Circulating blood FSH
    "FSH_pit",    # Pituitary FSH reservoir
    "E2",         # Circulating Estradiol
    "P4",         # Circulating Progesterone
    "InhB",       # Circulating Inhibin B
    "AF1",        # Early antral follicles
    "AF2",        # Antral follicles
    "AF3",        # Pre-ovulatory stage follicles
    "PrF",        # Pre-ovulatory follicles
]


def run_isolated_parameter_ablation(
    patient: HPOPatientValues,
    parameter_id: str,
    start_day: float = 0,
    end_day: float = 120,
    points: int = 1201,
) -> Dict[str, Any]:
    """
    Executes a single-parameter isolated ablation for any specific parameter:
      1. Simulates generic baseline (all parameters at published baseline).
      2. Simulates personalized baseline with ONLY the requested parameter personalized.
      3. Validates trajectory sanity and computes differences across all species.
    """
    twin = PatientDigitalTwin(patient)

    # 1. Generic unpersonalized baseline
    generic_sim = twin.run_baseline(
        start_day=start_day,
        end_day=end_day,
        points=points,
        personalized=False,
    )

    sbml_reg = get_sbml_parameter_registry()
    canonical_id = sbml_reg.resolve_id(parameter_id)
    base_val = sbml_reg.get_baseline_value(canonical_id)

    pers_val = twin.mapping.ode_parameters.get(canonical_id, base_val)

    # 2. Personalized simulation with isolated parameter
    personalized_sim = twin.run_baseline(
        start_day=start_day,
        end_day=end_day,
        points=points,
        personalized=False,
        parameter_overrides={canonical_id: pers_val},
    )

    # 3. Trajectory sanity validation
    sanity_passed, err_msg, warnings = validate_trajectory_sanity(
        personalized_sim.result,
        expected_points=points,
    )

    # 4. Trajectory differences
    full_diff = twin.compute_trajectory_difference(
        generic_sim.result,
        personalized_sim.result,
    )

    key_diff = {
        k: full_diff[k]
        for k in KEY_PHYSIOLOGICAL_SPECIES
        if k in full_diff
    }

    return {
        "patient_id": patient.patient_id,
        "mode": f"isolated_{canonical_id}",
        "parameter_id": canonical_id,
        "baseline_value": base_val,
        "personalized_value": pers_val,
        "delta_pct": (pers_val - base_val) / base_val * 100.0,
        "trajectory_shape": generic_sim.result.shape,
        "sanity_passed": sanity_passed,
        "sanity_error": err_msg,
        "sanity_warnings": warnings,
        "key_comparison": key_diff,
        "full_comparison": full_diff,
        "generic_simulation": generic_sim,
        "personalized_simulation": personalized_sim,
    }


def run_combined_core_5_ablation(
    patient: HPOPatientValues,
    start_day: float = 0,
    end_day: float = 120,
    points: int = 1201,
) -> Dict[str, Any]:
    """
    TC-A6: Executes combined Core 5-parameter personalized simulation:
      Maps p1, p17, p49, p51, p173 simultaneously against generic baseline.
    """
    twin = PatientDigitalTwin(patient)

    # 1. Generic unpersonalized baseline
    generic_sim = twin.run_baseline(
        start_day=start_day,
        end_day=end_day,
        points=points,
        personalized=False,
    )

    # 2. Personalized simulation with full core 5 parameters
    personalized_sim = twin.run_baseline(
        start_day=start_day,
        end_day=end_day,
        points=points,
        personalized=True,
    )

    # 3. Trajectory sanity validation
    sanity_passed, err_msg, warnings = validate_trajectory_sanity(
        personalized_sim.result,
        expected_points=points,
    )

    # 4. Trajectory differences
    full_diff = twin.compute_trajectory_difference(
        generic_sim.result,
        personalized_sim.result,
    )

    key_diff = {
        k: full_diff[k]
        for k in KEY_PHYSIOLOGICAL_SPECIES
        if k in full_diff
    }

    # Extract all 5 parameters
    sbml_reg = get_sbml_parameter_registry()
    core_params_summary = {}
    for pid in ["p1", "p17", "p49", "p51", "p173"]:
        b_val = sbml_reg.get_baseline_value(pid)
        p_val = twin.mapping.ode_parameters.get(pid, b_val)
        core_params_summary[pid] = {
            "baseline": b_val,
            "personalized": p_val,
            "delta_pct": (p_val - b_val) / b_val * 100.0,
        }

    return {
        "patient_id": patient.patient_id,
        "mode": "combined_core_5",
        "core_parameters": core_params_summary,
        "trajectory_shape": generic_sim.result.shape,
        "sanity_passed": sanity_passed,
        "sanity_error": err_msg,
        "sanity_warnings": warnings,
        "key_comparison": key_diff,
        "full_comparison": full_diff,
        "generic_simulation": generic_sim,
        "personalized_simulation": personalized_sim,
    }


def export_ablation_summary(
    ablation_result: Dict[str, Any],
    output_path: Path | str,
) -> Path:
    """Exports key species ablation deltas to a clean CSV summary."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for species, stats in ablation_result["key_comparison"].items():
        records.append({
            "species": species,
            "baseline_final": stats["baseline_final"],
            "personalized_final": stats["scenario_final"],
            "final_difference": stats["final_difference"],
            "max_absolute_difference": stats["maximum_absolute_difference"],
        })

    df = pd.DataFrame(records)
    df.to_csv(out_path, index=False)
    return out_path
