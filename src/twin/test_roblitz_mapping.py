"""
test_roblitz_mapping.py

Unit test and verification suite for Phase 7.4:
Verifying the two-layer bounded scaling function (tanh), hard clamps, missing data handling,
and audit logging.
"""

import sys
import math
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.twin.roblitz_mapping import (
    CLINICAL_REFERENCES,
    MappingAuditEntry,
    bounded_scale,
    map_parameter_value,
)
from src.HPO.parameters import get_authoritative_registry


def test_bounded_scale_identity():
    # TC-01a: When clinical value equals reference, scale factor must be exactly 1.0
    scale = bounded_scale(value=5.0, reference=5.0, max_delta=0.35, alpha=0.8)
    assert scale == 1.0, f"Expected 1.0, got {scale}"
    print("[OK] Test 1 (TC-01a): Identity verification - S(x_ref) == 1.000 exactly.")


def test_bounded_scale_soft_bounds():
    # TC-01b: Moderate perturbation
    scale_mod = bounded_scale(value=15.0, reference=5.0, max_delta=0.35, alpha=0.8)
    expected_mod = 1.0 + 0.35 * math.tanh(0.8 * (15.0 - 5.0) / 5.0)
    assert math.isclose(scale_mod, expected_mod, rel_tol=1e-6)
    assert 1.0 < scale_mod < 1.35
    print(f"     -> Moderate value (15.0 vs ref 5.0) -> scale={scale_mod:.4f}")

    # TC-01c: Extreme high value must asymptotically approach 1.0 + max_delta, never exceeding it
    scale_extreme_high = bounded_scale(value=50000.0, reference=5.0, max_delta=0.35, alpha=0.8)
    assert scale_extreme_high <= 1.35, f"Exceeded maximum upper bound: {scale_extreme_high}"
    assert math.isclose(scale_extreme_high, 1.35, rel_tol=1e-4)

    # TC-01d: Low value near zero must stay strictly positive and >= 1.0 - max_delta
    scale_zero = bounded_scale(value=0.0, reference=5.0, max_delta=0.35, alpha=0.8)
    assert scale_zero >= 0.65, f"Violated minimum lower bound: {scale_zero}"
    assert scale_zero > 0.0

    print("[OK] Test 2 (TC-01b, c, d): Soft bounds strictly confine perturbation within [0.65, 1.35].")


def test_bounded_scale_missing_and_invalid():
    # TC-01e: Missing, NaN, Inf must safely fall back to 1.0
    assert bounded_scale(value=None, reference=5.0) == 1.0
    assert bounded_scale(value=float("nan"), reference=5.0) == 1.0
    assert bounded_scale(value=float("inf"), reference=5.0) == 1.0

    # Negative values treated as 0.0 safely
    scale_neg = bounded_scale(value=-50.0, reference=5.0)
    assert scale_neg >= 0.65

    # Non-positive reference must raise ValueError
    try:
        bounded_scale(value=5.0, reference=0.0)
        assert False, "Expected ValueError for non-positive reference."
    except ValueError:
        pass

    print("[OK] Test 3 (TC-01e): Graceful fallback to 1.0 on None, NaN, and Inf.")


def test_bounded_scale_inverse_direction():
    # When direction = -1, higher value should decrease scale
    scale_inv_high = bounded_scale(value=10.0, reference=5.0, direction=-1, max_delta=0.35)
    assert scale_inv_high < 1.0, f"Expected scale < 1.0 for inverse direction, got {scale_inv_high}"

    scale_inv_low = bounded_scale(value=2.0, reference=5.0, direction=-1, max_delta=0.35)
    assert scale_inv_low > 1.0, f"Expected scale > 1.0 for inverse direction, got {scale_inv_low}"

    print("[OK] Test 4: Inverse direction (direction = -1) correctly inverts scaling response.")


def test_map_parameter_value_personalized():
    reg = get_authoritative_registry()
    base_p1 = reg.get("p1").baseline_value

    # Test mapping p1 with LH = 10.0 (reference = 5.0)
    final_val, audit = map_parameter_value(
        param_key="p1",
        clinical_value=10.0,
        reference_value=CLINICAL_REFERENCES["LH"],
        marker_name="LH",
        max_delta=0.35,
        alpha=0.8,
    )

    assert final_val > base_p1
    assert audit.status == "personalized"
    assert audit.parameter_id == "p1"
    assert audit.parameter_name == "b_syn_LH"
    assert audit.scale_factor > 1.0
    assert audit.final_value == final_val
    print(f"     -> Mapped p1 (LH=10.0): baseline={base_p1:.2f} -> final={final_val:.2f} [status={audit.status}]")
    print("[OK] Test 5: map_parameter_value correctly scales and personalizes p1.")


def test_map_parameter_value_missing_fallback():
    # TC-02: Missing observation handling and audit logging
    reg = get_authoritative_registry()
    base_p173 = reg.get("p173").baseline_value

    final_val, audit = map_parameter_value(
        param_key="b_IhB",
        clinical_value=None,
        reference_value=CLINICAL_REFERENCES["AMH"],
        marker_name="AMH",
    )

    assert final_val == base_p173
    assert audit.status == "missing_fallback"
    assert audit.scale_factor == 1.0
    assert "missing" in audit.reason
    print(f"     -> Missing marker fallback: {audit}")
    print("[OK] Test 6 (TC-02): Missing clinical observation defaults to baseline with explicit audit log.")


def test_map_parameter_value_hard_clamping():
    # Extreme scaling clamped by Layer 2 hard safety bounds
    reg = get_authoritative_registry()
    meta_p1 = reg.get("p1")

    # Manually pass huge max_delta = 5.0 to test Layer 2 hard clamping
    final_val, audit = map_parameter_value(
        param_key="p1",
        clinical_value=500.0,
        reference_value=5.0,
        marker_name="LH",
        max_delta=5.0,
    )

    assert final_val <= meta_p1.max_bound
    assert final_val == meta_p1.max_bound
    assert audit.status == "clamped"
    assert "clamped" in audit.reason
    print(f"     -> Hard safety clamp enforced: {audit.reason}")
    print("[OK] Test 7: Layer 2 hard safety clamping restricts extreme values to max_bound.")


def main():
    print("=" * 65)
    print("PHASE 7.4: BOUNDED PARAMETER MAPPING & AUDIT LOGGING VERIFICATION")
    print("=" * 65)

    test_bounded_scale_identity()
    test_bounded_scale_soft_bounds()
    test_bounded_scale_missing_and_invalid()
    test_bounded_scale_inverse_direction()
    test_map_parameter_value_personalized()
    test_map_parameter_value_missing_fallback()
    test_map_parameter_value_hard_clamping()

    print("=" * 65)
    print("ALL PHASE 7.4 TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    main()
