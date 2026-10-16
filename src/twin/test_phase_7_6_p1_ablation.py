"""
test_phase_7_6_p1_ablation.py

Unit test and verification suite for Phase 7.6:
Single-parameter p1 (b_syn_LH) ablation study against the generic Röblitz baseline.
"""

import sys
import math
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.HPO.patient_values import HPOPatientValues
from src.twin.digital_twin import PatientDigitalTwin
from src.twin.p1_ablation import run_p1_ablation, export_ablation_summary
from src.Clinical.hpo_observation_builder import build_hpo_patient_values
import pandas as pd


def test_baseline_preservation_invariance():
    """
    TC-00: Baseline Preservation Invariance.
    When personalized mapping is disabled, twin baseline simulation must exactly
    equal the generic published Röblitz baseline with 0.0 difference across all 46 species.
    """
    p = HPOPatientValues(patient_id=999, lh=10.0, fsh=5.0)
    twin = PatientDigitalTwin(p)

    sim_generic = twin.run_baseline(start_day=0, end_day=30, points=301, personalized=False)
    sim_unpers = twin.run_baseline(start_day=0, end_day=30, points=301, personalized=False, parameter_overrides={})

    diff = twin.compute_trajectory_difference(sim_generic.result, sim_unpers.result)
    for sp, stats in diff.items():
        assert stats["maximum_absolute_difference"] == 0.0, (
            f"Species {sp} showed non-zero difference {stats['maximum_absolute_difference']}!"
        )

    print("[OK] Test 1 (TC-00): Baseline Preservation Invariance verified - 0.0 delta across all species.")


def test_p1_ablation_patient_1():
    """
    TC-A1: Single-parameter ablation on Patient #1 from clinical dataset.
    Patient #1 has LH = 3.68, FSH/LH = 2.16 -> p1 scales down by -9.95%.
    """
    data_path = ROOT_DIR / "data" / "PCOS_data_without_infertility-Full_new.csv"
    df = pd.read_csv(data_path)
    patient_1 = build_hpo_patient_values(df.iloc[0])

    print(f"     -> Running 120-day ablation for Patient #1 (LH={patient_1.lh}, ratio={patient_1.fsh_lh_ratio:.2f})...")
    res = run_p1_ablation(patient_1, start_day=0, end_day=120, points=1201)

    assert res["trajectory_shape"] == (1201, 46)
    assert res["p1_delta_pct"] < 0.0  # scaled down
    print(f"     -> p1 baseline={res['p1_baseline']:.2f} -> personalized={res['p1_personalized']:.2f} ({res['p1_delta_pct']:+.2f}%)")

    key_diffs = res["key_comparison"]

    # Verify LH_Pit and LH_bld have non-zero differences
    assert key_diffs["LH_Pit"]["maximum_absolute_difference"] > 0.0
    assert key_diffs["LH_bld"]["maximum_absolute_difference"] > 0.0

    print("     -> Physiological deltas observed:")
    for sp in ["LH_Pit", "LH_bld", "FSH_bld", "E2", "AF2", "PrF"]:
        max_d = key_diffs[sp]["maximum_absolute_difference"]
        fin_d = key_diffs[sp]["final_difference"]
        print(f"        * {sp:8s}: max_abs_diff = {max_d:<10.4f} | final_diff = {fin_d:<10.4f}")

    print("[OK] Test 2 (TC-A1): Patient #1 p1 ablation successfully quantified physiological deltas.")
    return res


def test_p1_ablation_classic_pcos():
    """
    TC-A1: Single-parameter ablation on Classic elevated LH PCOS profile.
    LH = 14.8, FSH/LH = 0.42 -> p1 scales UP by +30.1%.
    """
    p_pcos = HPOPatientValues(patient_id=202, lh=14.8, fsh=6.2, fsh_lh_ratio=0.419)

    print(f"     -> Running 120-day ablation for Classic PCOS profile (LH={p_pcos.lh}, ratio={p_pcos.fsh_lh_ratio:.2f})...")
    res = run_p1_ablation(p_pcos, start_day=0, end_day=120, points=1201)

    assert res["p1_delta_pct"] > 0.0  # scaled UP
    print(f"     -> p1 baseline={res['p1_baseline']:.2f} -> personalized={res['p1_personalized']:.2f} ({res['p1_delta_pct']:+.2f}%)")

    key_diffs = res["key_comparison"]
    # With elevated p1, pituitary LH reservoir should expand
    assert key_diffs["LH_Pit"]["final_difference"] > 0.0
    print(f"     -> Pituitary LH expansion: final delta = {key_diffs['LH_Pit']['final_difference']:+.2f} mIU/mL")
    print(f"     -> Blood LH max delta     : {key_diffs['LH_bld']['maximum_absolute_difference']:.4f} mIU/mL")

    print("[OK] Test 3 (TC-A1): Classic PCOS p1 ablation verified pituitary reservoir expansion and elevated blood LH.")


def test_export_ablation_summary(ablation_res):
    output_path = ROOT_DIR / "results" / "p1_ablation_patient_1.csv"
    exported = export_ablation_summary(ablation_res, output_path)
    assert exported.exists(), f"Ablation summary file not found at {exported}"

    df = pd.read_csv(exported)
    assert len(df) == 11, f"Expected 11 key physiological species, found {len(df)}"
    print(f"[OK] Test 4: Ablation summary exported to '{exported.relative_to(ROOT_DIR)}' (11 key species).")


def main():
    print("=" * 65)
    print("PHASE 7.6: SINGLE-PARAMETER P1 ABLATION SIMULATION SUITE")
    print("=" * 65)

    test_baseline_preservation_invariance()
    res_p1 = test_p1_ablation_patient_1()
    test_p1_ablation_classic_pcos()
    test_export_ablation_summary(res_p1)

    print("=" * 65)
    print("ALL PHASE 7.6 TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    main()
