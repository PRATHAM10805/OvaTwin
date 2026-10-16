"""
p17_ablation.py

Phase 7.8: Dual-Gonadotropin Personalization & Ablation Engine.
Quantifies the exact trajectory impact of:
  1. Isolated parameter p17 (b_rel_FSH) ablation vs generic baseline.
  2. Joint dual-gonadotropin (p1 + p17) personalization vs generic baseline.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd

from src.HPO.patient_values import HPOPatientValues
from src.twin.digital_twin import PatientDigitalTwin
from src.HPO.sbml_parameters import get_sbml_parameter_registry
from src.twin.roblitz_mapping import map_p17_b_rel_FSH, map_p1_b_syn_LH
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


def run_p17_ablation(
    patient: HPOPatientValues,
    start_day: float = 0,
    end_day: float = 120,
    points: int = 1201,
) -> Dict[str, Any]:
    """
    Executes a single-parameter ablation for p17 (b_rel_FSH):
      1. Simulates generic baseline (all parameters at published baseline).
      2. Simulates personalized baseline with ONLY p17 personalized.
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

    base_p17 = get_sbml_parameter_registry().get_baseline_value("p17")
    p17_val = twin.mapping.ode_parameters.get("p17")
    if p17_val is None:
        p17_val, _ = map_p17_b_rel_FSH(patient)

    # 2. Personalized simulation with p17 only
    personalized_sim = twin.run_baseline(
        start_day=start_day,
        end_day=end_day,
        points=points,
        personalized=False,
        parameter_overrides={"p17": p17_val},
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
        "mode": "p17_isolated",
        "p17_baseline": base_p17,
        "p17_personalized": p17_val,
        "p17_delta_pct": (p17_val - base_p17) / base_p17 * 100.0,
        "trajectory_shape": generic_sim.result.shape,
        "sanity_passed": sanity_passed,
        "sanity_error": err_msg,
        "sanity_warnings": warnings,
        "key_comparison": key_diff,
        "full_comparison": full_diff,
        "generic_simulation": generic_sim,
        "personalized_simulation": personalized_sim,
    }


def run_joint_gonadotropin_ablation(
    patient: HPOPatientValues,
    start_day: float = 0,
    end_day: float = 120,
    points: int = 1201,
) -> Dict[str, Any]:
    """
    Executes joint dual-gonadotropin personalization (p1 + p17):
      1. Simulates generic baseline (all parameters at published baseline).
      2. Simulates personalized baseline with BOTH p1 and p17 personalized.
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

    base_p1 = get_sbml_parameter_registry().get_baseline_value("p1")
    base_p17 = get_sbml_parameter_registry().get_baseline_value("p17")

    p1_val = twin.mapping.ode_parameters.get("p1")
    if p1_val is None:
        p1_val, _ = map_p1_b_syn_LH(patient)

    p17_val = twin.mapping.ode_parameters.get("p17")
    if p17_val is None:
        p17_val, _ = map_p17_b_rel_FSH(patient)

    # 2. Joint personalized simulation (p1 + p17)
    personalized_sim = twin.run_baseline(
        start_day=start_day,
        end_day=end_day,
        points=points,
        personalized=False,
        parameter_overrides={"p1": p1_val, "p17": p17_val},
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
        "mode": "joint_p1_p17",
        "p1_baseline": base_p1,
        "p1_personalized": p1_val,
        "p1_delta_pct": (p1_val - base_p1) / base_p1 * 100.0,
        "p17_baseline": base_p17,
        "p17_personalized": p17_val,
        "p17_delta_pct": (p17_val - base_p17) / base_p17 * 100.0,
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
