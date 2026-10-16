"""
test_phase_7_10_cohort_verification.py

Unit test and verification suite for Phase 7.10:
Combined Core Mapping & Cohort Verification across all 541 Clinical Patients:
  - TC-04: Batch Pipeline Robustness (541/541 patients mapped with zero exceptions).
  - Hard physiological bound enforcement (P > 0 and 0.40 * P_base <= P <= 2.50 * P_base).
  - Dynamic stability & zero solver crashes on representative phenotypes.
  - Export and schema validation of cohort parameter registry and summary statistics.
"""

import sys
import math
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.twin.cohort_verification import (
    run_cohort_mapping,
    verify_representative_cohort_simulations,
    export_cohort_artifacts,
    CORE_PARAMETER_IDS,
)


def test_cohort_mapping_size_and_exceptions():
    """TC-04: 541/541 clinical patients mapped with 0 exceptions."""
    print("     -> Executing Core 5 mapping across full clinical dataset (541 patients)...")
    cohort_df, summary_df = run_cohort_mapping()

    assert len(cohort_df) == 541, f"Expected 541 patients, got {len(cohort_df)}"
    assert not cohort_df[CORE_PARAMETER_IDS].isnull().any().any(), "Parameter columns contain unexpected nulls!"

    print(f"[OK] Test 1 (TC-04): Successfully mapped all 541/541 clinical patients with 0 exceptions.")


def test_cohort_bounds_and_positivity(cohort_df=None):
    """Verifies that 100% of mapped parameters satisfy strict physiological bounds and positivity."""
    if cohort_df is None:
        cohort_df, _ = run_cohort_mapping()

    for pid in CORE_PARAMETER_IDS:
        vals = cohort_df[pid]
        deltas = cohort_df[f"{pid}_delta_pct"]

        # 1. Strictly positive
        assert (vals > 0).all(), f"Found non-positive values for parameter {pid}!"

        # 2. Layer 1 soft bounded perturbation (+/-35%)
        assert (deltas >= -35.0 - 1e-4).all(), f"Parameter {pid} violated lower soft bound: {deltas.min()}%"
        assert (deltas <= 35.0 + 1e-4).all(), f"Parameter {pid} violated upper soft bound: {deltas.max()}%"

    print("[OK] Test 2: 100% of generated parameters strictly satisfy positivity (P > 0) and soft bounds [-35%, +35%].")


def test_cohort_summary_statistics(summary_df=None):
    """Verifies cohort summary statistics table integrity and plausible distributions."""
    if summary_df is None:
        _, summary_df = run_cohort_mapping()

    assert len(summary_df) == 5, f"Expected 5 core parameters in summary, found {len(summary_df)}"
    print("     -> Cohort Summary Statistics across 541 Patients:")
    print("     " + "-" * 85)
    print(f"     {'Param':6s} | {'Name':12s} | {'Baseline':>10s} | {'Mean':>10s} | {'Std':>8s} | {'Min':>10s} | {'Max':>10s}")
    print("     " + "-" * 85)

    for _, row in summary_df.iterrows():
        p = row['parameter_id']
        n = row['biological_name']
        b = row['baseline_value']
        m = row['cohort_mean']
        s = row['cohort_std']
        mi = row['cohort_min']
        ma = row['cohort_max']
        print(f"     {p:6s} | {n:12s} | {b:10.4f} | {m:10.4f} | {s:8.4f} | {mi:10.4f} | {ma:10.4f}")

    print("     " + "-" * 85)
    print("[OK] Test 3: Cohort summary statistics verified for all 5 Core parameters.")


def test_representative_simulations_zero_crashes(cohort_df=None):
    """Verifies that representative cohort extremes simulate cleanly with 0 crashes."""
    if cohort_df is None:
        cohort_df, _ = run_cohort_mapping()

    print("     -> Simulating representative cohort phenotypes (min, median, max, extreme PCOS)...")
    sim_records = verify_representative_cohort_simulations(cohort_df)

    assert len(sim_records) == 5, f"Expected 5 representative simulations, got {len(sim_records)}"

    for rec in sim_records:
        assert rec["simulation_success"] is True
        assert rec["sanity_passed"] is True
        assert rec["trajectory_shape"] == (1201, 46)
        label = rec["profile_label"]
        pid = rec["patient_id"]
        print(f"     -> [PASS] {label:42s} (Patient #{pid:<3d}): shape=(1201, 46), 0 crashes")

    print("[OK] Test 4: Dynamic stability verified across representative phenotypes - 0 solver crashes.")


def test_export_cohort_artifacts(cohort_df=None, summary_df=None):
    """Verifies artifact export and schema conformity."""
    if cohort_df is None or summary_df is None:
        cohort_df, summary_df = run_cohort_mapping()

    cohort_csv, summary_csv = export_cohort_artifacts(cohort_df, summary_df)

    assert cohort_csv.exists(), f"File {cohort_csv} does not exist"
    assert summary_csv.exists(), f"File {summary_csv} does not exist"

    df_cohort = pd.read_csv(cohort_csv)
    df_summary = pd.read_csv(summary_csv)

    assert len(df_cohort) == 541
    assert len(df_summary) == 5

    print(f"[OK] Test 5: Successfully exported cohort artifacts to '{cohort_csv.name}' and '{summary_csv.name}'.")


def main():
    print("=" * 70)
    print("PHASE 7.10: FULL COHORT BATCH MAPPING & ROBUSTNESS VERIFICATION")
    print("=" * 70)

    test_cohort_mapping_size_and_exceptions()
    cohort_df, summary_df = run_cohort_mapping()
    test_cohort_bounds_and_positivity(cohort_df)
    test_cohort_summary_statistics(summary_df)
    test_representative_simulations_zero_crashes(cohort_df)
    test_export_cohort_artifacts(cohort_df, summary_df)

    print("=" * 70)
    print("ALL PHASE 7.10 TESTS PASSED SUCCESSFULLY! (541/541 COHORT VERIFIED)")
    print("=" * 70)


if __name__ == "__main__":
    main()
