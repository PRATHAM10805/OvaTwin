import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.HPO.model_runner import (
    RoblitzModelRunner,
)


def main():

    runner = RoblitzModelRunner()

    print("=" * 70)
    print("RÖBLITZ MODEL EXECUTION TEST")
    print("=" * 70)

    print("\nSPECIES")
    print("-" * 70)

    species = runner.get_species_ids()

    print(f"Number of species: {len(species)}")

    print("\nFirst 20 species:")

    for species_id in species[:20]:
        print(species_id)

    print("\nPARAMETERS")
    print("-" * 70)

    parameters = runner.get_parameters()

    print(
        f"Number of parameters: "
        f"{len(parameters)}"
    )

    print("\nFirst 20 parameters:")

    for parameter in parameters[:20]:
        print(parameter)

    print("\nINITIAL STATE")
    print("-" * 70)

    initial_state = runner.get_initial_state()

    print(
        f"Number of floating species: "
        f"{len(initial_state)}"
    )

    print(
        "Initial state loaded successfully."
    )

    print("\nSIMULATION")
    print("-" * 70)

    result = runner.simulate(
        start=0,
        end=120,
        points=1201,
    )

    print(
        f"Simulation shape: {result.shape}"
    )

    print(
        f"Simulation completed successfully."
    )

    print("\n" + "=" * 70)
    print("RÖBLITZ MODEL TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
    