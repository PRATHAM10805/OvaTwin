"""OvaTwin Phase 2: Clinical preprocessing and derived features."""

from pathlib import Path

import numpy as np
import pandas as pd

from .clinical_loader import load_clinical_data, validate_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "PCOS_data_without_infertility-Full_new.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "clinical_dataset_preprocessed.csv"
)


# These are project screening thresholds, not clinical
# diagnostic cutoffs. Suspicious observations are retained.
EXTREME_THRESHOLDS = {
    "FSH(mIU/mL)": 50,
    "LH(mIU/mL)": 50,
    "AMH(ng/mL)": 30,
    "PRL(ng/mL)": 100,
    "PRG(ng/mL)": 20,
    "TSH (mIU/L)": 20,
}


# Numeric measurements and encoded numeric fields.
NUMERIC_COLUMNS = [
    "Age (yrs)",
    "Weight (Kg)",
    "Height(Cm)",
    "BMI",
    "Waist:Hip Ratio",
    "Waist(inch)",
    "Hip(inch)",
    "Pulse rate(bpm)",
    "RR (breaths/min)",
    "Hb(g/dl)",
    "Cycle length(days)",
    "Marraige Status (Yrs)",
    "FSH(mIU/mL)",
    "LH(mIU/mL)",
    "I   beta-HCG(mIU/mL)",
    "II    beta-HCG(mIU/mL)",
    "AMH(ng/mL)",
    "PRL(ng/mL)",
    "PRG(ng/mL)",
    "TSH (mIU/L)",
    "Vit D3 (ng/mL)",
    "RBS(mg/dl)",
    "BP _Systolic (mmHg)",
    "BP _Diastolic (mmHg)",
    "Follicle No. (L)",
    "Follicle No. (R)",
    "Avg. F size (L) (mm)",
    "Avg. F size (R) (mm)",
    "Endometrium (mm)",
]


# Fields calculated from other measurements are audited
# separately rather than treated as primary measurements.
SOURCE_DERIVED_FIELDS = {
    "BMI": "BMI_source_raw",
    "Waist:Hip Ratio": "Waist_Hip_Ratio_source_raw",
    "FSH/LH": "FSH_LH_source_raw",
}


def _clean_columns(df):
    """Standardize column names without changing patient rows."""
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    return df


def _missing_mask(series):
    """Identify actual missing values and empty strings."""
    text = series.astype("string").str.strip()

    return (
        series.isna()
        | text.eq("").fillna(False)
    )


def _numeric(series):
    """Convert numeric values; malformed text becomes NaN."""
    return pd.to_numeric(series, errors="coerce")


def _finite(series):
    """Return True only for finite numeric values."""
    values = _numeric(series)

    return (
        values.notna()
        & np.isfinite(values.astype(float))
    )


def _safe_column_name(column):
    """Build readable names for generated quality flags."""
    name = "".join(
        char if char.isalnum() else "_"
        for char in column
    ).strip("_")

    while "__" in name:
        name = name.replace("__", "_")

    return name


def preprocess_clinical_data(df, raw_df=None):
    """
    Prepare clinical observations without deleting patients.

    The original source-derived fields are preserved in dedicated
    audit columns. Calculated features are stored separately.

    This output is a preprocessed clinical dataset, not yet the
    final feature matrix for training a machine-learning model.
    """
    df = _clean_columns(df).reset_index(drop=True)

    if raw_df is None:
        raw = df.copy()
    else:
        raw = _clean_columns(raw_df).reset_index(drop=True)

    if len(df) != len(raw):
        raise ValueError(
            "Loaded and raw datasets have different row counts."
        )

    # ---------------------------------------------------------
    # 1. Preserve source values for derived fields
    # ---------------------------------------------------------

    for source_column, audit_column in (
        SOURCE_DERIVED_FIELDS.items()
    ):
        if source_column in raw.columns:
            df[audit_column] = (
                raw[source_column]
                .astype("string")
                .reset_index(drop=True)
            )

    # ---------------------------------------------------------
    # 2. Convert numeric fields
    # ---------------------------------------------------------

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = _numeric(df[column])

    # ---------------------------------------------------------
    # 3. Recalculate BMI from weight and height
    # ---------------------------------------------------------

    if {
        "Weight (Kg)",
        "Height(Cm)",
    }.issubset(df.columns):

        weight = df["Weight (Kg)"]
        height = df["Height(Cm)"]

        valid_bmi = (
            _finite(weight)
            & _finite(height)
            & weight.gt(0)
            & height.gt(0)
        )

        df["computed_BMI"] = np.nan

        df.loc[valid_bmi, "computed_BMI"] = (
            weight.loc[valid_bmi]
            / (height.loc[valid_bmi] / 100) ** 2
        )

    # ---------------------------------------------------------
    # 4. Recalculate waist-to-hip ratio
    # ---------------------------------------------------------

    if {
        "Waist(inch)",
        "Hip(inch)",
    }.issubset(df.columns):

        waist = df["Waist(inch)"]
        hip = df["Hip(inch)"]

        valid_waist_hip = (
            _finite(waist)
            & _finite(hip)
            & waist.ge(0)
            & hip.gt(0)
        )

        df["computed_Waist_Hip_Ratio"] = np.nan

        df.loc[
            valid_waist_hip,
            "computed_Waist_Hip_Ratio",
        ] = (
            waist.loc[valid_waist_hip]
            / hip.loc[valid_waist_hip]
        )

    # ---------------------------------------------------------
    # 5. Calculate FSH/LH ratio safely
    # ---------------------------------------------------------

    if {
        "FSH(mIU/mL)",
        "LH(mIU/mL)",
    }.issubset(df.columns):

        fsh = df["FSH(mIU/mL)"]
        lh = df["LH(mIU/mL)"]

        valid_ratio = (
            _finite(fsh)
            & _finite(lh)
            & fsh.ge(0)
            & lh.gt(0)
        )

        df["FSH_LH_ratio"] = np.nan

        df.loc[valid_ratio, "FSH_LH_ratio"] = (
            fsh.loc[valid_ratio]
            / lh.loc[valid_ratio]
        )

        # Use a transformed version for experiments where
        # the ratio's extreme right skew is problematic.
        # The original ratio is not modified.
        df["log1p_FSH_LH_ratio"] = np.log1p(
            df["FSH_LH_ratio"]
        )

        df["flag_uncomputable_FSH_LH_ratio"] = (
            ~valid_ratio
        )

        df["flag_extreme_FSH_LH_ratio"] = (
            valid_ratio
            & df["FSH_LH_ratio"].gt(100)
        )

    # ---------------------------------------------------------
    # 6. Calculate total follicle count
    # ---------------------------------------------------------

    follicle_columns = [
        "Follicle No. (L)",
        "Follicle No. (R)",
    ]

    if set(follicle_columns).issubset(df.columns):

        left = df[follicle_columns[0]]
        right = df[follicle_columns[1]]

        valid_left = (
            _finite(left)
            & left.ge(0)
        )

        valid_right = (
            _finite(right)
            & right.ge(0)
        )

        both_valid = valid_left & valid_right

        df["total_follicle_count"] = np.nan

        df.loc[both_valid, "total_follicle_count"] = (
            left.loc[both_valid]
            + right.loc[both_valid]
        )

        df["flag_incomplete_follicle_count"] = (
            ~(valid_left & valid_right)
        )

    # ---------------------------------------------------------
    # 7. Calculate mean follicle size
    # ---------------------------------------------------------

    size_columns = [
        "Avg. F size (L) (mm)",
        "Avg. F size (R) (mm)",
    ]

    if set(size_columns).issubset(df.columns):

        left_size = df[size_columns[0]]
        right_size = df[size_columns[1]]

        left_valid = (
            _finite(left_size)
            & left_size.ge(0)
        )

        right_valid = (
            _finite(right_size)
            & right_size.ge(0)
        )

        sizes = pd.DataFrame(
            {
                "left": left_size.where(left_valid),
                "right": right_size.where(right_valid),
            },
            index=df.index,
        )

        # Calculate from available valid sides.
        # If both sides are unavailable, the result stays missing.
        df["mean_follicle_size_mm"] = (
            sizes.mean(axis=1)
        )

        df["flag_incomplete_follicle_size"] = (
            ~(left_valid & right_valid)
        )

    # ---------------------------------------------------------
    # 8. Flag source-format issues
    # ---------------------------------------------------------

    source_issue_flags = []

    for column in NUMERIC_COLUMNS:
        if column not in raw.columns:
            continue

        # BMI and waist/hip ratio are derived fields, so their
        # source-format problems are tracked separately below.
        if column in {"BMI", "Waist:Hip Ratio"}:
            continue

        source = raw[column]
        missing = _missing_mask(source)
        parsed = _numeric(source)

        malformed = (~missing) & parsed.isna()

        safe_name = _safe_column_name(column)
        flag_name = f"flag_nonnumeric_{safe_name}"

        df[flag_name] = malformed.to_numpy(dtype=bool)

        source_issue_flags.append(flag_name)

    for source_column in [
        "BMI",
        "Waist:Hip Ratio",
        "FSH/LH",
    ]:
        if source_column not in raw.columns:
            continue

        source = raw[source_column]
        missing = _missing_mask(source)
        parsed = _numeric(source)

        malformed = (~missing) & parsed.isna()

        safe_name = _safe_column_name(source_column)
        flag_name = f"flag_invalid_source_{safe_name}"

        df[flag_name] = malformed.to_numpy(dtype=bool)

        source_issue_flags.append(flag_name)

    # ---------------------------------------------------------
    # 9. Flag missing, negative, and non-finite measurements
    # ---------------------------------------------------------

    measurement_flag_columns = []

    derived_source_columns = {
        "BMI",
        "Waist:Hip Ratio",
        "FSH/LH",
    }

    for column in NUMERIC_COLUMNS:
        if column not in raw.columns:
            continue

        if column in derived_source_columns:
            continue

        source = raw[column]
        values = _numeric(source)

        safe_name = _safe_column_name(column)

        missing = _missing_mask(source)

        nonfinite = (
            values.notna()
            & ~np.isfinite(values.astype(float))
        )

        finite = (
            values.notna()
            & np.isfinite(values.astype(float))
        )

        negative = finite & values.lt(0)

        checks = {
            f"flag_missing_{safe_name}": missing,
            f"flag_nonfinite_{safe_name}": nonfinite,
            f"flag_negative_{safe_name}": negative,
        }

        for flag_name, mask in checks.items():
            df[flag_name] = mask.to_numpy(dtype=bool)
            measurement_flag_columns.append(flag_name)

        malformed_flag = f"flag_nonnumeric_{safe_name}"

        if malformed_flag in df.columns:
            measurement_flag_columns.append(malformed_flag)

    # ---------------------------------------------------------
    # 10. Flag extreme measurements
    # ---------------------------------------------------------

    extreme_flag_columns = []

    for column, threshold in EXTREME_THRESHOLDS.items():
        if column not in df.columns:
            continue

        values = df[column]

        valid = _finite(values)

        flag_name = (
            "flag_extreme_"
            + _safe_column_name(column)
        )

        # Only positive values above the review threshold are
        # included here. Negative measurements have separate flags.
        df[flag_name] = (
            valid & values.gt(threshold)
        )

        extreme_flag_columns.append(flag_name)

    df["has_extreme_value"] = (
        df[extreme_flag_columns].any(axis=1)
        if extreme_flag_columns
        else False
    )

    df["has_measurement_quality_issue"] = (
        df[measurement_flag_columns].any(axis=1)
        if measurement_flag_columns
        else False
    )

    df["has_source_format_issue"] = (
        df[source_issue_flags].any(axis=1)
        if source_issue_flags
        else False
    )

    # ---------------------------------------------------------
    # 11. Patient-ID integrity
    # ---------------------------------------------------------

    id_flags = []

    id_column = "Patient File No."

    if id_column in raw.columns:
        source_ids = raw[id_column]
        missing_ids = _missing_mask(source_ids)
        numeric_ids = _numeric(source_ids)

        finite_ids = (
            numeric_ids.notna()
            & np.isfinite(numeric_ids.astype(float))
        )

        invalid_ids = (~missing_ids) & (
            numeric_ids.isna()
            | (~finite_ids)
            | (
                finite_ids
                & numeric_ids.mod(1).ne(0)
            )
        )

        valid_ids = (
            (~missing_ids)
            & (~invalid_ids)
        )

        ids_for_duplicate_check = numeric_ids.where(
            valid_ids
        )

        duplicate_ids = (
            ids_for_duplicate_check.notna()
            & ids_for_duplicate_check.duplicated(
                keep=False
            )
        )

        id_checks = {
            "flag_missing_patient_id": missing_ids,
            "flag_invalid_patient_id": invalid_ids,
            "flag_duplicate_patient_id": duplicate_ids,
        }

        for flag_name, mask in id_checks.items():
            df[flag_name] = mask.to_numpy(dtype=bool)
            id_flags.append(flag_name)

    df["has_record_integrity_issue"] = (
        df[id_flags].any(axis=1)
        if id_flags
        else False
    )

    # ---------------------------------------------------------
    # 12. Preserve cycle fields without guessing their meaning
    # ---------------------------------------------------------

    df["cycle_encoding_requires_review"] = True

    # Keep the original ratio text in FSH_LH_source_raw.
    # Drop the corrupted source ratio from the downstream table.
    if "FSH/LH" in df.columns:
        df = df.drop(columns=["FSH/LH"])

    # ---------------------------------------------------------
    # 13. Overall quality summary
    # ---------------------------------------------------------

    df["has_quality_issue"] = df[
        [
            "has_measurement_quality_issue",
            "has_source_format_issue",
            "has_record_integrity_issue",
            "has_extreme_value",
        ]
    ].any(axis=1)

    return df


def print_preprocessing_summary(df):
    """Print row counts, derived features, and data-quality flags."""
    print("\n" + "=" * 70)
    print("OVATWIN — CLINICAL PREPROCESSING SUMMARY")
    print("=" * 70)

    print(f"\nPatients retained: {len(df)}")
    print(f"Columns after preprocessing: {len(df.columns)}")

    print("\nDerived feature availability:")

    for column in [
        "computed_BMI",
        "computed_Waist_Hip_Ratio",
        "FSH_LH_ratio",
        "log1p_FSH_LH_ratio",
        "total_follicle_count",
        "mean_follicle_size_mm",
    ]:
        if column in df.columns:
            print(
                f"{column:<32}: "
                f"{int(df[column].notna().sum())} / {len(df)}"
            )

    print("\nQuality summary:")

    for column in [
        "has_quality_issue",
        "has_measurement_quality_issue",
        "has_source_format_issue",
        "has_record_integrity_issue",
        "has_extreme_value",
        "flag_extreme_FSH_LH_ratio",
    ]:
        if column in df.columns:
            print(
                f"{column:<38}: "
                f"{int(df[column].sum())}"
            )

    print("\nMissing values in important measurements:")

    important_columns = [
        "Weight (Kg)",
        "Height(Cm)",
        "computed_BMI",
        "Waist(inch)",
        "Hip(inch)",
        "computed_Waist_Hip_Ratio",
        "FSH(mIU/mL)",
        "LH(mIU/mL)",
        "FSH_LH_ratio",
        "AMH(ng/mL)",
        "PRL(ng/mL)",
        "PRG(ng/mL)",
        "TSH (mIU/L)",
        "Follicle No. (L)",
        "Follicle No. (R)",
        "total_follicle_count",
        "mean_follicle_size_mm",
        "Endometrium (mm)",
    ]

    for column in important_columns:
        if column in df.columns:
            print(
                f"{column:<32}: "
                f"{int(df[column].isna().sum())}"
            )

    print("\nCycle variables:")
    print("- Cycle(R/I) is retained.")
    print("- Cycle length(days) is retained.")
    print("- Neither is assumed to represent literal cycle days.")


def save_preprocessed_data(df):
    """Save the preprocessed dataset without overwriting the source."""
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(OUTPUT_PATH, index=False)

    print("\nPreprocessed dataset saved:")
    print(OUTPUT_PATH.resolve())
    print(f"Rows retained: {len(df)}")


def main():
    """Run the clinical preprocessing pipeline."""
    print("=" * 70)
    print("OVATWIN PHASE 2")
    print("CLINICAL PREPROCESSING")
    print("=" * 70)

    raw_df = pd.read_csv(
        DATA_PATH,
        keep_default_na=False,
    )
    raw_df = _clean_columns(raw_df)

    df = load_clinical_data(DATA_PATH)
    validate_dataset(df)

    print(f"\nOriginal shape: {df.shape}")

    preprocessed = preprocess_clinical_data(
        df,
        raw_df=raw_df,
    )

    print_preprocessing_summary(preprocessed)
    save_preprocessed_data(preprocessed)


if __name__ == "__main__":
    main()