import sys
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.Clinical.hpo_observation_builder import build_hpo_patient_values
from src.twin.digital_twin import PatientDigitalTwin


DATA_PATH = ROOT_DIR / "data" / "PCOS_data_without_infertility-Full_new.csv"


def main():
    print("=" * 60)
    print("DIGITAL TWIN")
    print("=" * 60)

    # ---------------------------------------------------------
    # LOAD PATIENT
    # ---------------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    row = df.iloc[0]

    patient = build_hpo_patient_values(row)

    # ---------------------------------------------------------
    # CREATE DIGITAL TWIN
    # ---------------------------------------------------------

    twin = PatientDigitalTwin(patient)

    print("\nSTATUS")
    print("-" * 60)
    print(twin.get_status())

    print("\nSUMMARY")
    print("-" * 60)
    print(twin.summary())

    # ---------------------------------------------------------
    # BASELINE SIMULATION
    # ---------------------------------------------------------

    print("\nBASELINE SIMULATION")
    print("-" * 60)

    baseline = twin.run_baseline(
        start_day=0,
        end_day=120,
        points=1201,
    )

    print(f"Shape: {baseline.result.shape}")

    print("\nSTATUS AFTER BASELINE")
    print("-" * 60)
    print(twin.get_status())

    print("\nSUMMARY AFTER BASELINE")
    print("-" * 60)
    print(twin.summary())

    # ---------------------------------------------------------
    # WHAT-IF SCENARIO
    # ---------------------------------------------------------

    print("\nWHAT-IF SCENARIO")
    print("-" * 60)

    scenario = twin.run_scenario(
        name="test_parameter_change",
        parameter_changes={
            "b_syn_LH": 7309.91587614104 * 1.001,
        },
        start_day=0,
        end_day=120,
        points=1201,
    )

    print(f"Scenario name: {scenario.name}")
    print(f"Parameter changes: {scenario.parameter_changes}")
    print(f"Shape: {scenario.result.shape}")

    # ---------------------------------------------------------
    # FINAL STATE
    # ---------------------------------------------------------

    print("\nSTATUS AFTER SCENARIO")
    print("-" * 60)
    print(twin.get_status())

    print("\nFINAL SUMMARY")
    print("-" * 60)
    print(twin.summary())

    print("\n" + "=" * 60)
    print("DIGITAL TWIN SCENARIO COMPLETE")
    print("=" * 60)
    print(twin.summary())
    print("\nBASELINE VS SCENARIO")
    print("-" * 60)

    comparison = twin.compare_baseline_scenario(
        "test_parameter_change"
    )

    for variable, values in comparison.items():
        print(f"\n{variable}")
        print(f"  Baseline final : {values['baseline_final']}")
        print(f"  Scenario final : {values['scenario_final']}")
        print(f"  Difference     : {values['final_difference']}")
        print(
            f"  Max difference : "
            f"{values['maximum_absolute_difference']}"
        )
if __name__ == "__main__":
    main()