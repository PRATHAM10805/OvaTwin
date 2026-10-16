"""
test_phase_7_5_p1_mapping.py

Unit test and verification suite for Phase 7.5:
Single-parameter mapping of patient LH and FSH/LH ratio to Röblitz parameter p1 (b_syn_LH).
"""

import sys
import math
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.HPO.patient_values import HPOPatientValues
from src.twin.roblitz_mapping import map_p1_b_syn_LH
from src.twin.parameter_mapping import create_mapping
from src.HPO.parameters import get_authoritative_registry


def test_p1_median_patient():
    # Patient with median values (LH=5.0, FSH/LH=1.2)
    p_med = HPOPatientValues(patient_id=101, lh=5.0, fsh=6.0, fsh_lh_ratio=1.2)
    p1_val, audit = map_p1_b_syn_LH(p_med)

    base_p1 = get_authoritative_registry().get("p1").baseline_value
    assert math.isclose(p1_val, base_p1, rel_tol=1e-5), f"Expected {base_p1}, got {p1_val}"
    assert audit.scale_factor == 1.0
    assert audit.status == "personalized"
    print(f"     -> Median patient (LH=5.0, ratio=1.2): p1 = {p1_val:.4f} (scale={audit.scale_factor:.3f})")
    print("[OK] Test 1: Median patient produces exact published baseline p1 (7309.9159).")


def test_p1_classic_pcos_patient():
    # Classic PCOS profile: elevated LH (14.5), low FSH/LH ratio (0.45)
    p_pcos = HPOPatientValues(patient_id=102, lh=14.5, fsh=6.5, fsh_lh_ratio=0.448)
    p1_val, audit = map_p1_b_syn_LH(p_pcos)

    base_p1 = get_authoritative_registry().get("p1").baseline_value
    assert p1_val > base_p1
    assert 1.20 <= audit.scale_factor <= 1.35
    assert audit.status == "personalized"
    print(f"     -> Classic PCOS patient (LH=14.5, ratio=0.45): p1 = {p1_val:.4f} (scale={audit.scale_factor:.3f})")
    print("[OK] Test 2: Classic PCOS patient scales p1 UP within the expected +20% to +35% window.")


def test_p1_hypogonadotropic_patient():
    # Hypogonadotropic / low LH profile (LH=1.5, ratio=2.8)
    p_low = HPOPatientValues(patient_id=103, lh=1.5, fsh=4.2, fsh_lh_ratio=2.8)
    p1_val, audit = map_p1_b_syn_LH(p_low)

    base_p1 = get_authoritative_registry().get("p1").baseline_value
    assert p1_val < base_p1
    assert 0.65 <= audit.scale_factor <= 0.90
    assert audit.status == "personalized"
    print(f"     -> Low LH patient (LH=1.5, ratio=2.8): p1 = {p1_val:.4f} (scale={audit.scale_factor:.3f})")
    print("[OK] Test 3: Low LH patient scales p1 DOWN safely within [0.65, 0.90].")


def test_p1_missing_lh_fallback():
    # Missing LH observation (None)
    p_none = HPOPatientValues(patient_id=104, lh=None, fsh_lh_ratio=None)
    p1_val, audit = map_p1_b_syn_LH(p_none)

    base_p1 = get_authoritative_registry().get("p1").baseline_value
    assert p1_val == base_p1
    assert audit.status == "missing_fallback"
    assert audit.scale_factor == 1.0
    assert "missing" in audit.reason
    print(f"     -> Missing LH patient: {audit}")
    print("[OK] Test 4: Missing LH gracefully defaults to baseline with audit log.")


def test_p1_extreme_outlier_clamping():
    # Extreme outlier (LH=2000 mIU/mL, ratio=0.001)
    p_extreme = HPOPatientValues(patient_id=105, lh=2000.0, fsh_lh_ratio=0.001)
    p1_val, audit = map_p1_b_syn_LH(p_extreme)

    base_p1 = get_authoritative_registry().get("p1").baseline_value
    max_bound = get_authoritative_registry().get("p1").max_bound

    assert p1_val <= max_bound
    # Soft bound limits total_delta to 0.35 -> scale_net = 1.35
    assert math.isclose(audit.scale_factor, 1.35, rel_tol=1e-3)
    print(f"     -> Extreme outlier (LH=2000): p1 = {p1_val:.4f} (scale={audit.scale_factor:.3f})")
    print("[OK] Test 5: Extreme clinical outlier strictly soft-bounded to +35% and safely within hard limits.")


def test_twin_mapping_integration():
    p = HPOPatientValues(patient_id=1, lh=7.5, fsh=5.0, fsh_lh_ratio=0.67)

    # 1. Personalized mapping enabled
    mapping_pers = create_mapping(p, personalize_ode=True)
    assert "p1" in mapping_pers.ode_parameters
    assert "b_syn_LH" in mapping_pers.ode_parameters
    assert mapping_pers.ode_parameters["p1"] == mapping_pers.ode_parameters["b_syn_LH"]
    assert len(mapping_pers.mapping_audit) >= 1
    print(f"     -> Twin mapping ode_parameters: {mapping_pers.ode_parameters}")

    # 2. Personalized mapping disabled (baseline preservation)
    mapping_unpers = create_mapping(p, personalize_ode=False)
    assert len(mapping_unpers.ode_parameters) == 0
    assert len(mapping_unpers.mapping_audit) == 0

    print("[OK] Test 6: PatientTwinMapping integration verified for both personalized and unpersonalized modes.")


def main():
    print("=" * 65)
    print("PHASE 7.5: SINGLE-PARAMETER P1 (b_syn_LH) MAPPING VERIFICATION")
    print("=" * 65)

    test_p1_median_patient()
    test_p1_classic_pcos_patient()
    test_p1_hypogonadotropic_patient()
    test_p1_missing_lh_fallback()
    test_p1_extreme_outlier_clamping()
    test_twin_mapping_integration()

    print("=" * 65)
    print("ALL PHASE 7.5 TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    main()
