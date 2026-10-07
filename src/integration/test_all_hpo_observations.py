import sys
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.Clinical.hpo_observation_builder import (
    build_hpo_patient_values,
)

from src.HPO.validation import (
    validate_patient_values,
)


DATA_PATH = ROOT_DIR / "data" / "PCOS_data_without_infertility-Full_new.csv"


def main():

    df = pd.read_csv(DATA_PATH)

    valid_count = 0
    invalid_count = 0

    missing_summary = {}

    for _, row in df.iterrows():

        patient = build_hpo_patient_values(row)

        validation = validate_patient_values(
            patient
        )

        if validation["valid"]:
            valid_count += 1

        else:
            invalid_count += 1

            for field in validation["missing"]:

                missing_summary[field] = (
                    missing_summary.get(field, 0) + 1
                )

    print("\nHPO OBSERVATION VALIDATION")
    print("=" * 60)

    print(
        f"Total patients : {len(df)}"
    )

    print(
        f"Valid patients : {valid_count}"
    )

    print(
        f"Invalid patients : {invalid_count}"
    )

    print("\nMissing required observations")
    print("-" * 60)

    if missing_summary:

        for field, count in missing_summary.items():

            print(
                f"{field}: {count}"
            )

    else:

        print("None")


if __name__ == "__main__":
    main()