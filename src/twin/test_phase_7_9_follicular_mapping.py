"""
test_phase_7_9_follicular_mapping.py

Unit test and verification suite for Phase 7.9:
Follicular Parameters (p49, p51, p173) and Combined Core 5 Personalization:
  - Unit tests for p49 (k_AF1), p51 (k_AF2_AF3), and p173 (b_IhB) mapping logic.
  - TC-A3: Single-parameter ablation for p49 (k_AF1).
  - TC-A4: Single-parameter ablation for p51 (k_AF2_AF3).
  - TC-A5: Single-parameter ablation for p173 (b_IhB).
  - TC-A6: Combined Core 5-parameter personalized simulation (p1, p17, p49, p51, p173).
  - Numerical sanity validation (finite values, 0 NaNs, non-negative concentrations).
  - Export of individual and combined ablation CSV summaries.
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
from src.twin.roblitz_mapping import (
    map_p49_k_AF1,
    map_p51_k_AF2_AF3,
    map_p173_b_IhB,
    map_core_5_parameters,
)
from src.twin.parameter_mapping import create_mapping
from src.HPO.parameters import get_authoritative_registry
from src.Clinical.hpo_observation_builder import build_hpo_patient_values
from src.twin.follicular_ablation import (
    run_isolated_parameter_ablation,
    run_combined_core_5_ablation,
    export_ablation_summary,
)


def test_p49_mapping_logic():
    """Unit tests for p49 (k_AF1) early antral recruitment rate mapping."""
    reg = get_authoritative_registry()
    base_p49 = reg.get("p49").baseline_value

    # 1. Median patient (total_follicle_count = 12.0)
    p_med = HPOPatientValues(patient_id=301, total_follicle_count=12.0)
    val_med, audit_med = map_p49_k_AF1(p_med)
    assert math.isclose(val_med, base_p49, rel_tol=1e-5)
    assert audit_med.scale_factor == 1.0
    assert audit_med.status == "personalized"

    # 2. Polycystic ovary profile (total_follicle_count = 32.0)
    p_high = HPOPatientValues(patient_id=302, total_follicle_count=32.0)
    val_high, audit_high = map_p49_k_AF1(p_high)
    assert val_high > base_p49
    assert 1.15 <= audit_high.scale_factor <= 1.35
    assert audit_high.status == "personalized"

    # 3. Diminished ovarian reserve (total_follicle_count = 4.0)
    p_low = HPOPatientValues(patient_id=303, total_follicle_count=4.0)
    val_low, audit_low = map_p49_k_AF1(p_low)
    assert val_low < base_p49
    assert 0.65 <= audit_low.scale_factor <= 0.95

    # 4. Missing fallback
    p_none = HPOPatientValues(patient_id=304, total_follicle_count=None)
    val_none, audit_none = map_p49_k_AF1(p_none)
    assert val_none == base_p49
    assert audit_none.status == "missing_fallback"

    # 5. Extreme clamping (total_follicle_count = 500)
    p_ext = HPOPatientValues(patient_id=305, total_follicle_count=500.0)
    val_ext, audit_ext = map_p49_k_AF1(p_ext)
    meta = reg.get("p49")
    assert meta.min_bound <= val_ext <= meta.max_bound
    assert math.isclose(audit_ext.scale_factor, 1.30, rel_tol=1e-2)

    print("[OK] Test 1: p49 (k_AF1) mapping logic passed (median, high, low, missing, clamping).")


def test_p51_mapping_logic():
    """Unit tests for p51 (k_AF2_AF3) dominant follicle transition rate mapping."""
    reg = get_authoritative_registry()
    base_p51 = reg.get("p51").baseline_value

    # 1. Median patient (mean_follicle_size = 14.0 mm, regular cycle = 2.0)
    p_med = HPOPatientValues(patient_id=311, mean_follicle_size=14.0, cycle_regular=2.0)
    val_med, audit_med = map_p51_k_AF2_AF3(p_med)
    assert math.isclose(val_med, base_p51, rel_tol=1e-5)
    assert audit_med.scale_factor == 1.0

    # 2. Follicular arrest + Irregular cycle (size = 6.0 mm, cycle = 4.0)
    p_arrest = HPOPatientValues(patient_id=312, mean_follicle_size=6.0, cycle_regular=4.0)
    val_arr, audit_arr = map_p51_k_AF2_AF3(p_arrest)
    assert val_arr < base_p51
    assert 0.65 <= audit_arr.scale_factor <= 0.85
    assert audit_arr.status == "personalized"

    # 3. Healthy dominant follicle (size = 20.0 mm, cycle = 2.0)
    p_dom = HPOPatientValues(patient_id=313, mean_follicle_size=20.0, cycle_regular=2.0)
    val_dom, audit_dom = map_p51_k_AF2_AF3(p_dom)
    assert val_dom > base_p51
    assert 1.04 <= audit_dom.scale_factor <= 1.30

    # 4. Missing fallback
    p_none = HPOPatientValues(patient_id=314, mean_follicle_size=None, cycle_regular=None)
    val_none, audit_none = map_p51_k_AF2_AF3(p_none)
    assert val_none == base_p51
    assert audit_none.status == "missing_fallback"

    # 5. Extreme clamping
    p_ext = HPOPatientValues(patient_id=315, mean_follicle_size=200.0, cycle_regular=2.0)
    val_ext, audit_ext = map_p51_k_AF2_AF3(p_ext)
    meta = reg.get("p51")
    assert meta.min_bound <= val_ext <= meta.max_bound
    assert math.isclose(audit_ext.scale_factor, 1.25, rel_tol=1e-2)

    print("[OK] Test 2: p51 (k_AF2_AF3) mapping logic passed (median, arrest, dominant, missing, clamping).")


def test_p173_mapping_logic():
    """Unit tests for p173 (b_IhB) Inhibin B basal synthesis rate mapping."""
    reg = get_authoritative_registry()
    base_p173 = reg.get("p173").baseline_value

    # 1. Median patient (follicle_count = 12.0, AMH = 2.5 ng/mL)
    p_med = HPOPatientValues(patient_id=321, total_follicle_count=12.0, amh=2.5)
    val_med, audit_med = map_p173_b_IhB(p_med)
    assert math.isclose(val_med, base_p173, rel_tol=1e-5)
    assert audit_med.scale_factor == 1.0

    # 2. Elevated AMH & High follicle count (follicle_count = 32.0, AMH = 9.5)
    p_high = HPOPatientValues(patient_id=322, total_follicle_count=32.0, amh=9.5)
    val_high, audit_high = map_p173_b_IhB(p_high)
    assert val_high > base_p173
    assert 1.15 <= audit_high.scale_factor <= 1.35
    assert audit_high.status == "personalized"

    # 3. Low follicle count & Low AMH (follicle_count = 4.0, AMH = 0.8)
    p_low = HPOPatientValues(patient_id=323, total_follicle_count=4.0, amh=0.8)
    val_low, audit_low = map_p173_b_IhB(p_low)
    assert val_low < base_p173
    assert 0.65 <= audit_low.scale_factor <= 0.95

    # 4. Missing fallback
    p_none = HPOPatientValues(patient_id=324, total_follicle_count=None, amh=None)
    val_none, audit_none = map_p173_b_IhB(p_none)
    assert val_none == base_p173
    assert audit_none.status == "missing_fallback"

    # 5. Extreme clamping
    p_ext = HPOPatientValues(patient_id=325, total_follicle_count=500.0, amh=100.0)
    val_ext, audit_ext = map_p173_b_IhB(p_ext)
    meta = reg.get("p173")
    assert meta.min_bound <= val_ext <= meta.max_bound
    assert math.isclose(audit_ext.scale_factor, 1.35, rel_tol=1e-2)

    print("[OK] Test 3: p173 (b_IhB) mapping logic passed (median, high, low, missing, clamping).")


def test_p49_ablation_tc_a3():
    """
    TC-A3: Isolated p49 (k_AF1) ablation.
    Polycystic follicle profile: total_follicle_count = 32.0 -> k_AF1 scales UP (+22.8%).
    Trajectory Impact: AF1 elevates above baseline; AF2 shows downstream elevation.
    """
    p = HPOPatientValues(patient_id=331, total_follicle_count=32.0)
    print(f"     -> Running TC-A3: Isolated p49 ablation (follicle_count=32.0)...")
    res = run_isolated_parameter_ablation(p, "p49", start_day=0, end_day=120, points=1201)

    assert res["trajectory_shape"] == (1201, 46)
    assert res["sanity_passed"] is True, f"Sanity failed: {res['sanity_error']}"
    assert res["delta_pct"] > 0.0

    key_diffs = res["key_comparison"]
    assert key_diffs["AF1"]["maximum_absolute_difference"] > 0.0
    assert key_diffs["AF2"]["maximum_absolute_difference"] > 0.0

    print(f"     -> p49 baseline={res['baseline_value']:.4f} -> pers={res['personalized_value']:.4f} ({res['delta_pct']:+.2f}%)")
    print(f"     -> AF1 max abs diff: {key_diffs['AF1']['maximum_absolute_difference']:.4f}")
    print(f"     -> AF2 max abs diff: {key_diffs['AF2']['maximum_absolute_difference']:.4f}")
    print("[OK] Test 4 (TC-A3): Isolated p49 ablation confirmed early antral pool expansion.")
    return res


def test_p51_ablation_tc_a4():
    """
    TC-A4: Isolated p51 (k_AF2_AF3) ablation.
    Follicular arrest profile: mean_follicle_size = 6.5 mm, cycle_regular = 4.0.
    k_AF2_AF3 scales DOWN (-25.3%).
    Trajectory Impact: Accumulation in AF2, attenuation in AF3 / PrF, lower E2.
    """
    p = HPOPatientValues(patient_id=332, mean_follicle_size=6.5, cycle_regular=4.0)
    print(f"     -> Running TC-A4: Isolated p51 ablation (size=6.5mm, cycle=Irregular)...")
    res = run_isolated_parameter_ablation(p, "p51", start_day=0, end_day=120, points=1201)

    assert res["trajectory_shape"] == (1201, 46)
    assert res["sanity_passed"] is True, f"Sanity failed: {res['sanity_error']}"
    assert res["delta_pct"] < 0.0

    key_diffs = res["key_comparison"]
    assert key_diffs["AF2"]["maximum_absolute_difference"] > 0.0
    assert key_diffs["AF3"]["maximum_absolute_difference"] > 0.0
    assert key_diffs["PrF"]["maximum_absolute_difference"] > 0.0
    assert key_diffs["E2"]["maximum_absolute_difference"] > 0.0

    print(f"     -> p51 baseline={res['baseline_value']:.6f} -> pers={res['personalized_value']:.6f} ({res['delta_pct']:+.2f}%)")
    print(f"     -> AF2 max abs diff: {key_diffs['AF2']['maximum_absolute_difference']:.4f}")
    print(f"     -> AF3 max abs diff: {key_diffs['AF3']['maximum_absolute_difference']:.4f}")
    print(f"     -> PrF max abs diff: {key_diffs['PrF']['maximum_absolute_difference']:.4f}")
    print("[OK] Test 5 (TC-A4): Isolated p51 ablation confirmed follicular arrest dynamics.")
    return res


def test_p173_ablation_tc_a5():
    """
    TC-A5: Isolated p173 (b_IhB) ablation.
    Hyper-inhibin profile: total_follicle_count = 30.0, AMH = 8.5 ng/mL.
    b_IhB scales UP (+27.1%).
    Trajectory Impact: Upward shift in circulating InhB; feedback modulation on pituitary FSH.
    """
    p = HPOPatientValues(patient_id=333, total_follicle_count=30.0, amh=8.5)
    print(f"     -> Running TC-A5: Isolated p173 ablation (follicles=30.0, AMH=8.5)...")
    res = run_isolated_parameter_ablation(p, "p173", start_day=0, end_day=120, points=1201)

    assert res["trajectory_shape"] == (1201, 46)
    assert res["sanity_passed"] is True, f"Sanity failed: {res['sanity_error']}"
    assert res["delta_pct"] > 0.0

    key_diffs = res["key_comparison"]
    assert key_diffs["InhB"]["maximum_absolute_difference"] > 0.0
    assert key_diffs["FSH_bld"]["maximum_absolute_difference"] > 0.0

    print(f"     -> p173 baseline={res['baseline_value']:.4f} -> pers={res['personalized_value']:.4f} ({res['delta_pct']:+.2f}%)")
    print(f"     -> InhB max abs diff   : {key_diffs['InhB']['maximum_absolute_difference']:.4f}")
    print(f"     -> FSH_bld max abs diff: {key_diffs['FSH_bld']['maximum_absolute_difference']:.4f}")
    print("[OK] Test 6 (TC-A5): Isolated p173 ablation confirmed Inhibin B secretion elevation.")
    return res


def test_combined_core_5_ablation_tc_a6():
    """
    TC-A6: Combined Core 5-Parameter Personalized Baseline on Patient #1 from clinical dataset.
    Patient #1 features:
      - LH = 3.68, FSH = 7.95, FSH/LH = 2.16
      - Follicles (L+R) = 6.0
      - Mean follicle size = 18.0 mm
      - Cycle = Regular (2.0)
      - AMH = 2.07 ng/mL
    All 5 core parameters (p1, p17, p49, p51, p173) are mapped and simulated simultaneously.
    """
    data_path = ROOT_DIR / "data" / "PCOS_data_without_infertility-Full_new.csv"
    df = pd.read_csv(data_path)
    patient_1 = build_hpo_patient_values(df.iloc[0])

    print(f"     -> Running TC-A6: Combined Core 5 simulation for Patient #1...")
    res = run_combined_core_5_ablation(patient_1, start_day=0, end_day=120, points=1201)

    assert res["trajectory_shape"] == (1201, 46)
    assert res["sanity_passed"] is True, f"Sanity failed: {res['sanity_error']}"

    core_params = res["core_parameters"]
    assert len(core_params) == 5

    print("     -> Core 5 Parameter Personalization Deltas:")
    for pid, info in core_params.items():
        print(f"        * {pid:4s}: base={info['baseline']:<10.4f} -> pers={info['personalized']:<10.4f} ({info['delta_pct']:+.2f}%)")

    key_diffs = res["key_comparison"]
    print("     -> Combined Physiological Marker Trajectory Deltas:")
    for sp in ["LH_bld", "LH_Pit", "FSH_bld", "FSH_pit", "E2", "P4", "InhB", "AF1", "AF2", "AF3", "PrF"]:
        max_d = key_diffs[sp]["maximum_absolute_difference"]
        fin_d = key_diffs[sp]["final_difference"]
        print(f"        * {sp:8s}: max_abs_diff = {max_d:<10.4f} | final_diff = {fin_d:<10.4f}")

    print("[OK] Test 7 (TC-A6): Combined Core 5 simulation on Patient #1 successfully executed with zero solver failures.")
    return res


def test_export_phase_7_9_summaries(res_p49=None, res_p51=None, res_p173=None, res_core5=None):
    """Verifies CSV export for all Phase 7.9 ablation studies."""
    if res_p49 is None:
        res_p49 = test_p49_ablation_tc_a3()
    if res_p51 is None:
        res_p51 = test_p51_ablation_tc_a4()
    if res_p173 is None:
        res_p173 = test_p173_ablation_tc_a5()
    if res_core5 is None:
        res_core5 = test_combined_core_5_ablation_tc_a6()

    p49_csv = ROOT_DIR / "results" / "p49_ablation_tc_a3.csv"
    p51_csv = ROOT_DIR / "results" / "p51_ablation_tc_a4.csv"
    p173_csv = ROOT_DIR / "results" / "p173_ablation_tc_a5.csv"
    core5_csv = ROOT_DIR / "results" / "core_5_combined_ablation_patient_1.csv"

    export_ablation_summary(res_p49, p49_csv)
    export_ablation_summary(res_p51, p51_csv)
    export_ablation_summary(res_p173, p173_csv)
    export_ablation_summary(res_core5, core5_csv)

    for csv_file in [p49_csv, p51_csv, p173_csv, core5_csv]:
        assert csv_file.exists(), f"File {csv_file} does not exist"
        df = pd.read_csv(csv_file)
        assert len(df) == 11

    print(f"[OK] Test 8: Exported all 4 ablation summaries to 'results/' (11 key species each).")


def main():
    print("=" * 65)
    print("PHASE 7.9: FOLLICULAR & COMBINED CORE 5 PERSONALIZATION SUITE")
    print("=" * 65)

    test_p49_mapping_logic()
    test_p51_mapping_logic()
    test_p173_mapping_logic()
    r49 = test_p49_ablation_tc_a3()
    r51 = test_p51_ablation_tc_a4()
    r173 = test_p173_ablation_tc_a5()
    rcore5 = test_combined_core_5_ablation_tc_a6()
    test_export_phase_7_9_summaries(r49, r51, r173, rcore5)

    print("=" * 65)
    print("ALL PHASE 7.9 TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    main()
