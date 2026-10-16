"""
p1_ablation.py

Phase 7.6: Single-parameter ablation simulation engine.
Quantifies the exact trajectory impact of personalizing parameter p1 (b_syn_LH)
against the published Röblitz 2013 generic baseline.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd

from src.HPO.patient_values import HPOPatientValues
from src.twin.digital_twin import PatientDigitalTwin
from src.HPO.sbml_parameters import get_sbml_parameter_registry

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


def run_p1_ablation(
    patient: HPOPatientValues,
    start_day: float = 0,
    end_day: float = 120,
    points: int = 1201,
) -> Dict[str, Any]:
    """
    Executes a single-parameter ablation for p1 (b_syn_LH):
      1. Simulates generic baseline (p1 = 7309.9159, all baseline parameters).
      2. Simulates personalized baseline with ONLY p1 personalized.
      3. Computes trajectory differences across all 46 species.
    """
    twin = PatientDigitalTwin(patient)

    # 1. Generic unpersonalized baseline
    generic_sim = twin.run_baseline(
        start_day=start_day,
        end_day=end_day,
        points=points,
        personalized=False,
    )

    p1_personalized_val = twin.mapping.ode_parameters.get(
        "p1",
        get_sbml_parameter_registry().get_baseline_value("p1"),
    )

    # 2. Personalized baseline with p1 only
    personalized_sim = twin.run_baseline(
        start_day=start_day,
        end_day=end_day,
        points=points,
        personalized=True,
        parameter_overrides={"p1": p1_personalized_val},
    )

    # 3. Compute full trajectory comparison
    full_comparison = twin.compute_trajectory_difference(
        generic_sim.result,
        personalized_sim.result,
    )

    # 4. Filter key physiological markers
    key_comparison = {
        k: full_comparison[k]
        for k in KEY_PHYSIOLOGICAL_SPECIES
        if k in full_comparison
    }

    base_p1 = get_sbml_parameter_registry().get_baseline_value("p1")

    return {
        "patient_id": patient.patient_id,
        "p1_baseline": base_p1,
        "p1_personalized": p1_personalized_val,
        "p1_delta_pct": (p1_personalized_val - base_p1) / base_p1 * 100.0,
        "trajectory_shape": generic_sim.result.shape,
        "key_comparison": key_comparison,
        "full_comparison": full_comparison,
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
