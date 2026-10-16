"""
test_phase_7_7_simulation_guard.py

Unit test and verification suite for Phase 7.7:
Trajectory sanity validation, numerical stability checks, and explicit solver error capture.
"""

import sys
import numpy as np
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.twin.simulation_guard import (
    SimulationResult,
    safe_simulate,
    validate_trajectory_sanity,
)
from src.HPO.baseline_simulation import run_baseline_simulation


def test_successful_baseline_simulation():
    """
    Test 1: Full 120-day baseline simulation passes all sanity checks.
    """
    res = safe_simulate(start_day=0, end_day=120, points=1201)

    assert res.success is True, f"Expected success=True, got {res.error_message}"
    assert res.error_message is None
    assert res.trajectory is not None
    assert res.shape == (1201, 46)

    # Confirm zero NaN or Inf
    arr = np.asarray(res.trajectory)
    assert not np.isnan(arr).any()
    assert not np.isinf(arr).any()

    print("[OK] Test 1: Full 120-day baseline passes all numerical sanity checks (1201x46, 0 NaN, 0 Inf).")


def test_parameter_override_simulation():
    """
    Test 2: Safe simulate with personalized parameter override.
    """
    res = safe_simulate(
        parameter_overrides={"p1": 8500.0},
        start_day=0,
        end_day=60,
        points=601,
    )

    assert res.success is True
    assert "p1" in res.parameters_used
    assert res.parameters_used["p1"] == 8500.0
    assert res.shape == (601, 46)

    print(f"     -> Parameters used recorded: {res.parameters_used}")
    print("[OK] Test 2: Personalized parameter override safely executed and logged.")


def test_explicit_error_capture_no_silent_fallback():
    """
    Test 3 (TC-03): Explicit error capture on solver / setup failure.
    System MUST NOT silently substitute another baseline or mask the error.
    """
    # 1. Trigger solver error via invalid points (-1)
    res_bad_points = safe_simulate(start_day=0, end_day=120, points=-1)

    assert res_bad_points.success is False, "Expected success=False for invalid simulation parameters."
    assert res_bad_points.trajectory is None
    assert res_bad_points.error_message is not None
    assert "cannot be less than 2" in res_bad_points.error_message.lower() or "failure" in res_bad_points.error_message.lower()
    print(f"     -> Captured error correctly: {res_bad_points.error_message}")

    # 2. Trigger parameter error via non-existent parameter
    res_bad_param = safe_simulate(parameter_overrides={"invalid_fake_param_xyz": 999.0})
    assert res_bad_param.success is False
    assert res_bad_param.trajectory is None
    assert "Unknown Röblitz parameter" in res_bad_param.error_message or "Failed to set" in res_bad_param.error_message
    print(f"     -> Captured parameter error: {res_bad_param.error_message}")

    print("[OK] Test 3 (TC-03): Explicit error capture verified - zero silent fallbacks on failure.")


def test_trajectory_validator_corruptions():
    """
    Test 4: Trajectory validator catches artificial NaN, Inf, and unphysical negative concentrations.
    """
    # Create mock trajectory array of shape (100, 46)
    clean_arr = np.ones((100, 46))
    clean_arr[:, 0] = np.linspace(0, 10, 100)  # time

    # 1. Clean passes
    is_sane, err, _ = validate_trajectory_sanity(clean_arr, expected_points=100)
    assert is_sane is True
    assert err is None

    # 2. NaN corruption
    nan_arr = clean_arr.copy()
    nan_arr[50, 5] = np.nan
    is_sane, err, _ = validate_trajectory_sanity(nan_arr, expected_points=100)
    assert is_sane is False
    assert "NaN" in err

    # 3. Inf corruption
    inf_arr = clean_arr.copy()
    inf_arr[50, 5] = np.inf
    is_sane, err, _ = validate_trajectory_sanity(inf_arr, expected_points=100)
    assert is_sane is False
    assert "Infinite" in err

    # 4. Extreme unphysical negative concentration (-5.0)
    neg_arr = clean_arr.copy()
    neg_arr[50, 5] = -5.0
    is_sane, err, _ = validate_trajectory_sanity(neg_arr, expected_points=100)
    assert is_sane is False
    assert "negative" in err.lower()

    # 5. Minor numerical integration noise (-0.05) should warn, not fail
    noise_arr = clean_arr.copy()
    noise_arr[50, 5] = -0.05
    is_sane, err, warnings = validate_trajectory_sanity(noise_arr, expected_points=100)
    assert is_sane is True
    assert len(warnings) > 0

    print("[OK] Test 4: Trajectory sanity validator correctly catches NaN, Inf, and unphysical negative states.")


def test_baseline_simulation_integration():
    """
    Test 5: Integration with run_baseline_simulation.
    """
    sim = run_baseline_simulation(patient_id=1, start_day=0, end_day=10, points=101)
    assert sim.simulation_result is not None
    assert sim.simulation_result.success is True
    assert sim.result.shape == (101, 46)

    print("[OK] Test 5: run_baseline_simulation successfully exposes SimulationResult.")


def main():
    print("=" * 65)
    print("PHASE 7.7: TRAJECTORY SANITY & SOLVER ERROR CAPTURE VERIFICATION")
    print("=" * 65)

    test_successful_baseline_simulation()
    test_parameter_override_simulation()
    test_explicit_error_capture_no_silent_fallback()
    test_trajectory_validator_corruptions()
    test_baseline_simulation_integration()

    print("=" * 65)
    print("ALL PHASE 7.7 TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    main()
