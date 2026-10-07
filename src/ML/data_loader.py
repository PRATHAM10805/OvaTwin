import pandas as pd
from .config import DATA_PATH, TARGET_COLUMN


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{DATA_PATH}'.\n"
            f"Please place the PCOS questionnaire dataset at '{DATA_PATH}' as described in 'data/README.md'."
        )
    df = pd.read_csv(DATA_PATH)

    print("=" * 50)
    print("DATASET INFORMATION")
    print("=" * 50)

    print(f"Shape: {df.shape}")
    print(f"Target: {TARGET_COLUMN}")

    print("\nMissing values:")
    print(df.isnull().sum().sum())

    print("\nTarget distribution:")
    print(df[TARGET_COLUMN].value_counts())

    return df


def split_features_target(df):
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return X, y 