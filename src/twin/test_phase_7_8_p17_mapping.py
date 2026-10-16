"""
test_phase_7_8_p17_mapping.py

Unit test and verification suite for Phase 7.8:
Dual-Gonadotropin Personalization:
  - Unit tests for p17 (b_rel_FSH) mapping logic.
  - Isolated p17 ablation simulation (TC-A2).
  - Joint dual-gonadotropin (p1 + p17) personalization simulation.
  - Numerical trajectory sanity validation.
  - Export of ablation delta summaries.
"""

import sys
import math
from pathlib import Path
import pandas as pd

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.HPO.patient_values import HPOPatientValues
from src.twin.roblitz_mapping import map_p17_b_rel_FSH, map_p1_b_syn_LH
from src.twin.parameter_mapping import create_mapping
from src.HPO.parameters import get_authoritative_registry
from src.Clinical.hpo_observation_builder import build_hpo_patient_values
from src.twin.p17_ablation import (
    run_p17_ablation,
    run_joint_gonadotropin_ablation,
    export_ablation_summary,
)


def test_p17_median_patient():
    """Median clinical profile (FSH=5.5, FSH/LH=1.2) yields published baseline p17."""
    p_med = HPOPatientValues(patient_id=201, fsh=5.5, lh=4.58, fsh_lh_ratio=1.2)
    p17_val, audit = map_p17_b_rel_FSH(p_med)

    base_p17 = get_authoritative_registry().get("p17").baseline_value
    assert math.isclose(p17_val, base_p17, rel_tol=1e-5), f"Expected {base_p17}, got {p17_val}"
    assert audit.scale_factor == 1.0
    assert audit.status == "personalized"
    print(f"     -> Median patient (FSH=5.5, ratio=1.2): p17 = {p17_val:.6f} (scale={audit.scale_factor:.3f})")
    print("[OK] Test 1: Median patient produces exact published baseline p17 (0.057002).")


def test_p17_low_fsh_patient():
    """PCOS / Low FSH profile (FSH=2.8, ratio=0.45) scales p17 down safely."""
    p_low = HPOPatientValues(patient_id=202, fsh=2.8, lh=6.2, fsh_lh_ratio=0.452)
    p17_val, audit = map_p17_b_rel_FSH(p_low)

    base_p17 = get_authoritative_registry().get("p17").baseline_value
    assert p17_val < base_p17
    assert 0.65 <= audit.scale_factor <= 0.90
    assert audit.status == "personalized"
    print(f"     -> Low FSH patient (FSH=2.8, ratio=0.45): p17 = {p17_val:.6f} (scale={audit.scale_factor:.3f})")
    print("[OK] Test 2: Low FSH patient scales p17 DOWN safely within [0.65, 0.90].")


def test_p17_high_fsh_patient():
    """High FSH profile (FSH=12.0, ratio=2.5) scales p17 up safely."""
    p_high = HPOPatientValues(patient_id=203, fsh=12.0, lh=4.8, fsh_lh_ratio=2.5)
    p17_val, audit = map_p17_b_rel_FSH(p_high)

    base_p17 = get_authoritative_registry().get("p17").baseline_value
    assert p17_val > base_p17
    assert 1.15 <= audit.scale_factor <= 1.35
    assert audit.status == "personalized"
    print(f"     -> High FSH patient (FSH=12.0, ratio=2.5): p17 = {p17_val:.6f} (scale={audit.scale_factor:.3f})")
    print("[OK] Test 3: High FSH patient scales p17 UP safely within [1.15, 1.35].")


def test_p17_missing_fsh_fallback():
    """Missing FSH observation gracefully defaults to baseline with audit log."""
    p_none = HPOPatientValues(patient_id=204, fsh=None, fsh_lh_ratio=None)
    p17_val, audit = map_p17_b_rel_FSH(p_none)

    base_p17 = get_authoritative_registry().get("p17").baseline_value
    assert p17_val == base_p17
    assert audit.status == "missing_fallback"
    assert audit.scale_factor == 1.0
    assert "missing" in audit.reason
    print(f"     -> Missing FSH patient audit: {audit}")
    print("[OK] Test 4: Missing FSH gracefully defaults to baseline with audit log.")


def test_p17_extreme_outlier_clamping():
    """Extreme clinical outlier strictly soft-bounded to max +35% and within hard limits."""
    p_extreme = HPOPatientValues(patient_id=205, fsh=500.0, fsh_lh_ratio=50.0)
    p17_val, audit = map_p17_b_rel_FSH(p_extreme)

    meta_p17 = get_authoritative_registry().get("p17")
    assert meta_p17.min_bound <= p17_val <= meta_p17.max_bound
    assert math.isclose(audit.scale_factor, 1.35, rel_tol=1e-3)
    print(f"     -> Extreme outlier (FSH=500): p17 = {p17_val:.6f} (scale={audit.scale_factor:.3f})")
    print("[OK] Test 5: Extreme clinical outlier strictly soft-bounded to +35% and clamped within hard limits.")


def test_p17_isolated_ablation_patient_1():
    """
    TC-A2: Isolated p17 single-parameter ablation on Patient #1 from clinical dataset.
    Patient #1 has FSH = 7.95, FSH/LH = 2.16 -> p17 scales UP by +14.6%.
    """
    data_path = ROOT_DIR / "data" / "PCOS_data_without_infertility-Full_new.csv"
    df = pd.read_csv(data_path)
    patient_1 = build_hpo_patient_values(df.iloc[0])

    print(f"     -> Running isolated p17 ablation for Patient #1 (FSH={patient_1.fsh}, ratio={patient_1.fsh_lh_ratio:.2f})...")
    res = run_p17_ablation(patient_1, start_day=0, end_day=120, points=1201)

    assert res["trajectory_shape"] == (1201, 46)
    assert res["sanity_passed"] is True, f"Trajectory sanity failed: {res['sanity_error']}"
    assert res["p17_delta_pct"] > 0.0  # scaled up
    print(f"     -> p17 baseline={res['p17_baseline']:.6f} -> personalized={res['p17_personalized']:.6f} ({res['p17_delta_pct']:+.2f}%)")

    key_diffs = res["key_comparison"]
    # Verify FSH_bld has non-zero difference
    assert key_diffs["FSH_bld"]["maximum_absolute_difference"] > 0.0

    print("     -> Isolated p17 physiological deltas observed:")
    for sp in ["FSH_bld", "FSH_pit", "LH_bld", "E2", "AF1", "AF2", "PrF"]:
        max_d = key_diffs[sp]["maximum_absolute_difference"]
        fin_d = key_diffs[sp]["final_difference"]
        print(f"        * {sp:8s}: max_abs_diff = {max_d:<10.4f} | final_diff = {fin_d:<10.4f}")

    print("[OK] Test 6 (TC-A2): Patient #1 isolated p17 ablation quantified FSH pathway deltas cleanly.")
    return res


def test_joint_gonadotropin_ablation_patient_1():
    """
    Dual-gonadotropin joint ablation (p1 + p17) on Patient #1.
    Patient #1:
      - LH = 3.68  -> p1 scales down by -9.95%
      - FSH = 7.95 -> p17 scales up by +14.61%
    Simulates combined pituitary synthesis/release dynamics.
    """
    data_path = ROOT_DIR / "data" / "PCOS_data_without_infertility-Full_new.csv"
    df = pd.read_csv(data_path)
    patient_1 = build_hpo_patient_values(df.iloc[0])

    print(f"     -> Running joint p1+p17 dual-gonadotropin simulation for Patient #1...")
    res = run_joint_gonadotropin_ablation(patient_1, start_day=0, end_day=120, points=1201)

    assert res["trajectory_shape"] == (1201, 46)
    assert res["sanity_passed"] is True, f"Sanity failed: {res['sanity_error']}"
    assert res["p1_delta_pct"] < 0.0
    assert res["p17_delta_pct"] > 0.0

    print(f"     -> p1 : baseline={res['p1_baseline']:.2f} -> personalized={res['p1_personalized']:.2f} ({res['p1_delta_pct']:+.2f}%)")
    print(f"     -> p17: baseline={res['p17_baseline']:.6f} -> personalized={res['p17_personalized']:.6f} ({res['p17_delta_pct']:+.2f}%)")

    key_diffs = res["key_comparison"]
    assert key_diffs["LH_bld"]["maximum_absolute_difference"] > 0.0
    assert key_diffs["FSH_bld"]["maximum_absolute_difference"] > 0.0

    print("     -> Joint dual-gonadotropin deltas observed:")
    for sp in ["LH_bld", "LH_Pit", "FSH_bld", "FSH_pit", "E2", "P4", "InhB", "AF1", "AF2", "AF3", "PrF"]:
        max_d = key_diffs[sp]["maximum_absolute_difference"]
        fin_d = key_diffs[sp]["final_difference"]
        print(f"        * {sp:8s}: max_abs_diff = {max_d:<10.4f} | final_diff = {fin_d:<10.4f}")

    print("[OK] Test 7: Joint dual-gonadotropin (p1 + p17) simulation passed sanity and captured dual axis modulation.")
    return res


def test_export_ablation_summaries(p17_res=None, joint_res=None):
    """Verifies CSV export for isolated p17 and joint p1+p17 summaries."""
    if p17_res is None:
        p17_res = test_p17_isolated_ablation_patient_1()
    if joint_res is None:
        joint_res = test_joint_gonadotropin_ablation_patient_1()

    p17_csv = ROOT_DIR / "results" / "p17_ablation_patient_1.csv"
    joint_csv = ROOT_DIR / "results" / "p1_p17_joint_ablation_patient_1.csv"

    export_ablation_summary(p17_res, p17_csv)
    export_ablation_summary(joint_res, joint_csv)

    assert p17_csv.exists(), f"File {p17_csv} does not exist"
    assert joint_csv.exists(), f"File {joint_csv} does not exist"

    df_p17 = pd.read_csv(p17_csv)
    df_joint = pd.read_csv(joint_csv)

    assert len(df_p17) == 11
    assert len(df_joint) == 11

    print(f"[OK] Test 8: Exported ablation summaries to '{p17_csv.relative_to(ROOT_DIR)}' and '{joint_csv.relative_to(ROOT_DIR)}'.")


def main():
    print("=" * 65)
    print("PHASE 7.8: DUAL-GONADOTROPIN (P17 + P1) PERSONALIZATION SUITE")
    print("=" * 65)

    test_p17_median_patient()
    test_p17_low_fsh_patient()
    test_p17_high_fsh_patient()
    test_p17_missing_fsh_fallback()
    test_p17_extreme_outlier_clamping()
    p17_res = test_p17_isolated_ablation_patient_1()
    joint_res = test_joint_gonadotropin_ablation_patient_1()
    test_export_ablation_summaries(p17_res, joint_res)

    print("=" * 65)
    print("ALL PHASE 7.8 TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    main()
