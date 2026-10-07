import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.HPO.baseline_simulation import (
    run_baseline_simulation,
)


def main():

    print("=" * 70)
    print("OVATWIN BASELINE HPO SIMULATION")
    print("=" * 70)

    simulation = run_baseline_simulation(
        patient_id=1,
        start_day=0,
        end_day=120,
        points=1201,
    )

    print("\nSimulation information")
    print("-" * 70)

    print(
        f"Patient ID : {simulation.patient_id}"
    )

    print(
        f"Time       : "
        f"{simulation.start_day} -> "
        f"{simulation.end_day} days"
    )

    print(
        f"Points     : {simulation.points}"
    )

    print(
        f"Shape      : {simulation.result.shape}"
    )

    print("\nVariables")
    print("-" * 70)

    for name in simulation.result.colnames:
        print(name)

    print("\n" + "=" * 70)
    print("BASELINE SIMULATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()