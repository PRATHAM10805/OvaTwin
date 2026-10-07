import sys
from pathlib import Path
import pandas as pd

# Ensure project root is in sys.path when running directly
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.Clinical.hpo_observation_builder import (
    build_hpo_patient_values,
)

from src.twin.parameter_mapping import (
    create_mapping,
)


DATA_PATH = ROOT_DIR / "data" / "PCOS_data_without_infertility-Full_new.csv"


def main():

    df = pd.read_csv(DATA_PATH)

    patient = build_hpo_patient_values(
        df.iloc[0]
    )

    mapping = create_mapping(patient)

    print("\nPATIENT")
    print("=" * 60)
    print(patient)

    print("\nHORMONES")
    print("=" * 60)
    for key, value in mapping.hormones.items():
        print(f"{key}: {value}")

    print("\nFOLLICULAR OBSERVATIONS")
    print("=" * 60)
    for key, value in mapping.follicular.items():
        print(f"{key}: {value}")

    print("\nUTERINE OBSERVATIONS")
    print("=" * 60)
    for key, value in mapping.uterine.items():
        print(f"{key}: {value}")

    print("\nCYCLE OBSERVATIONS")
    print("=" * 60)
    for key, value in mapping.cycle.items():
        print(f"{key}: {value}")

    print("\nANTHROPOMETRIC OBSERVATIONS")
    print("=" * 60)
    for key, value in mapping.anthropometric.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
