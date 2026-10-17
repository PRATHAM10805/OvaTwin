"""Flag clinical-data quality issues without deleting patient records."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.Clinical.clinical_loader import (
    load_clinical_data,
    validate_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "PCOS_data_without_infertility-Full_new.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "clinical_dataset_flagged.csv"
)


# Project review thresholds only.
# These are NOT validated clinical diagnostic cutoffs.
EXTREME_THRESHOLDS = {
    "FSH(mIU/mL)": 50,
    "LH(mIU/mL)": 50,
    "AMH(ng/mL)": 30,
    "PRL(ng/mL)": 100,
    "PRG(ng/mL)": 20,
    "TSH (mIU/L)": 20,
}


NUMERIC_MEASUREMENT_COLUMNS = [
    "Age (yrs)",
    "Weight (Kg)",
    "Height(Cm)",
    "Pulse rate(bpm)",
    "RR (breaths/min)",
    "Hb(g/dl)",
    "Cycle length(days)",
    "Marraige Status (Yrs)",
    "FSH(mIU/mL)",
    "LH(mIU/mL)",
    "I   beta-HCG(mIU/mL)",
    "II    beta-HCG(mIU/mL)",
    "Hip(inch)",
    "Waist(inch)",
    "TSH (mIU/L)",
    "AMH(ng/mL)",
    "PRL(ng/mL)",
    "Vit D3 (ng/mL)",
    "PRG(ng/mL)",
    "RBS(mg/dl)",
    "BP _Systolic (mmHg)",
    "BP _Diastolic (mmHg)",
    "Follicle No. (L)",
    "Follicle No. (R)",
    "Avg. F size (L) (mm)",
    "Avg. F size (R) (mm)",
    "Endometrium (mm)",
]


def _clean_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names."""
    frame = frame.copy()
    frame.columns = frame.columns.astype(str).str.strip()
    return frame


def _missing_mask(series: pd.Series) -> pd.Series:
    """
    Identify actual missing values.

    Nulls and whitespace-only strings count as missing.
    Strings such as '#NAME?' remain malformed values,
    not missing values.
    """
    text = series.astype("string").str.strip()

    return (
        series.isna()
        | text.eq("").fillna(False)
    )


def _numeric(series: pd.Series) -> pd.Series:
    """Convert values to numeric; unparseable text becomes NaN."""
    return pd.to_numeric(series, errors="coerce")


def _flag_name(prefix: str, column: str) -> str:
    """Create a safe column name for a Boolean quality flag."""
    safe = "".join(
        character if character.isalnum() else "_"
        for character in column
    ).strip("_")

    while "__" in safe:
        safe = safe.replace("__", "_")

    return f"flag_{prefix}_{safe}"


def create_quality_flags(
    df: pd.DataFrame,
    raw_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Return all patient rows with additional quality flags.

    Distinguishes:
    - Missing measurements
    - Malformed numeric text
    - Positive or negative infinity
    - Negative numeric measurements
    - Values exceeding project screening thresholds
    - Invalid or duplicate patient IDs
    - Malformed spreadsheet-derived fields

    Pass raw_df directly from the CSV so malformed values such as
    '#NAME?' are not confused with missing values introduced by
    numeric conversion in the clinical loader.

    This function does not delete patient rows.
    """
    result = _clean_columns(df).reset_index(drop=True)

    raw_source = raw_df if raw_df is not None else df
    raw = _clean_columns(raw_source).reset_index(drop=True)

    if len(result) != len(raw):
        raise ValueError(
            "Loaded and raw datasets have different row counts; "
            "cannot align quality flags."
        )

    # Build flags separately, then concatenate them once.
    # This avoids repeated DataFrame column insertion.
    flags: dict[str, pd.Series] = {}

    measurement_flags: list[str] = []
    record_flags: list[str] = []
    source_flags: list[str] = []
    extreme_flags: list[str] = []

    # ---------------------------------------------------------
    # 1. Primary numeric measurement quality flags
    # ---------------------------------------------------------
    for column in NUMERIC_MEASUREMENT_COLUMNS:
        if column not in raw.columns:
            continue

        source = raw[column]

        missing = _missing_mask(source)
        values = _numeric(source)

        nonnumeric = (~missing) & values.isna()

        nonfinite = (
            values.notna()
            & ~np.isfinite(values.astype(float))
        )

        finite = (
            values.notna()
            & np.isfinite(values.astype(float))
        )

        checks = (
            ("missing_measurement", missing),
            ("nonnumeric_measurement", nonnumeric),
            ("nonfinite", nonfinite),
            ("negative", finite & values.lt(0)),
        )

        for prefix, mask in checks:
            name = _flag_name(prefix, column)

            flags[name] = (
                mask.astype(bool).reset_index(drop=True)
            )

            measurement_flags.append(name)

    # ---------------------------------------------------------
    # 2. Project screening thresholds
    # ---------------------------------------------------------
    # These are review flags, not clinical diagnoses.
    for column, threshold in EXTREME_THRESHOLDS.items():
        if column not in raw.columns:
            continue

        values = _numeric(raw[column])

        finite = (
            values.notna()
            & np.isfinite(values.astype(float))
        )

        name = _flag_name("extreme", column)

        flags[name] = (
            finite & values.gt(threshold)
        ).astype(bool).reset_index(drop=True)

        extreme_flags.append(name)

    # ---------------------------------------------------------
    # 3. Patient ID integrity
    # ---------------------------------------------------------
    if "Patient File No." in raw.columns:
        id_source = raw["Patient File No."]

        id_missing = _missing_mask(id_source)
        ids = _numeric(id_source)

        ids_finite = (
            ids.notna()
            & np.isfinite(ids.astype(float))
        )

        id_invalid = (~id_missing) & (
            ids.isna()
            | (~ids_finite)
            | (
                ids_finite
                & ids.mod(1).ne(0)
            )
        )

        valid_id = (~id_missing) & (~id_invalid)
        valid_ids = ids.where(valid_id)

        id_duplicate = (
            valid_ids.notna()
            & valid_ids.duplicated(keep=False)
        )

        id_checks = (
            ("flag_missing_patient_id", id_missing),
            ("flag_invalid_patient_id", id_invalid),
            ("flag_duplicate_patient_id", id_duplicate),
        )

        for name, mask in id_checks:
            flags[name] = (
                mask.astype(bool).reset_index(drop=True)
            )

            record_flags.append(name)

    # ---------------------------------------------------------
    # 4. Source-derived BMI and waist/hip ratio quality
    # ---------------------------------------------------------
    # Preserve the original source text in separate columns so
    # formula errors can be audited after numeric conversion.
    derived_fields = (
        ("BMI", "BMI"),
        ("Waist:Hip Ratio", "waist_hip_ratio"),
    )

    for column, label in derived_fields:
        if column not in raw.columns:
            continue

        source = raw[column]

        missing = _missing_mask(source)
        parsed = _numeric(source)

        malformed = (~missing) & parsed.isna()

        original_name = f"source_{label}_original"

        flags[original_name] = (
            source.astype("string").reset_index(drop=True)
        )

        checks = (
            (f"missing_source_{label}", missing),
            (f"invalid_source_{label}", malformed),
        )

        for prefix, mask in checks:
            name = f"flag_{prefix}"

            flags[name] = (
                mask.astype(bool).reset_index(drop=True)
            )

            source_flags.append(name)

    # ---------------------------------------------------------
    # 5. Original FSH/LH ratio quality
    # ---------------------------------------------------------
    # The original spreadsheet ratio is not trusted.
    # Recalculate it from the primary FSH and LH measurements.
    if "FSH/LH" in raw.columns:
        ratio_source = raw["FSH/LH"]

        ratio_missing = _missing_mask(ratio_source)
        ratio_values = _numeric(ratio_source)

        ratio_malformed = (
            (~ratio_missing)
            & ratio_values.isna()
        )

        ratio_checks = (
            ("flag_missing_source_ratio", ratio_missing),
            ("flag_invalid_source_ratio", ratio_malformed),
        )

        for name, mask in ratio_checks:
            flags[name] = (
                mask.astype(bool).reset_index(drop=True)
            )

            source_flags.append(name)

    # ---------------------------------------------------------
    # 6. Recalculate BMI from weight and height
    # ---------------------------------------------------------
    if {
        "Weight (Kg)",
        "Height(Cm)",
    }.issubset(raw.columns):

        weight = _numeric(raw["Weight (Kg)"])
        height = _numeric(raw["Height(Cm)"])

        valid_bmi = (
            weight.notna()
            & height.notna()
            & np.isfinite(weight.astype(float))
            & np.isfinite(height.astype(float))
            & weight.gt(0)
            & height.gt(0)
        )

        computed_bmi = pd.Series(
            np.nan,
            index=raw.index,
            dtype=float,
        )

        computed_bmi.loc[valid_bmi] = (
            weight.loc[valid_bmi]
            / (
                (height.loc[valid_bmi] / 100) ** 2
            )
        )

        flags["computed_BMI"] = (
            computed_bmi.reset_index(drop=True)
        )

    # ---------------------------------------------------------
    # 7. Recalculate waist/hip ratio
    # ---------------------------------------------------------
    if {
        "Waist(inch)",
        "Hip(inch)",
    }.issubset(raw.columns):

        waist = _numeric(raw["Waist(inch)"])
        hip = _numeric(raw["Hip(inch)"])

        valid_waist_hip = (
            waist.notna()
            & hip.notna()
            & np.isfinite(waist.astype(float))
            & np.isfinite(hip.astype(float))
            & waist.ge(0)
            & hip.gt(0)
        )

        computed_waist_hip = pd.Series(
            np.nan,
            index=raw.index,
            dtype=float,
        )

        computed_waist_hip.loc[valid_waist_hip] = (
            waist.loc[valid_waist_hip]
            / hip.loc[valid_waist_hip]
        )

        flags["computed_Waist_Hip_Ratio"] = (
            computed_waist_hip.reset_index(drop=True)
        )

    # ---------------------------------------------------------
    # 8. Recalculate FSH/LH ratio
    # ---------------------------------------------------------
    if {
        "FSH(mIU/mL)",
        "LH(mIU/mL)",
    }.issubset(raw.columns):

        fsh = _numeric(raw["FSH(mIU/mL)"])
        lh = _numeric(raw["LH(mIU/mL)"])

        valid_ratio = (
            fsh.notna()
            & lh.notna()
            & np.isfinite(fsh.astype(float))
            & np.isfinite(lh.astype(float))
            & lh.gt(0)
            & fsh.ge(0)
        )

        computed_ratio = pd.Series(
            np.nan,
            index=raw.index,
            dtype=float,
        )

        computed_ratio.loc[valid_ratio] = (
            fsh.loc[valid_ratio]
            / lh.loc[valid_ratio]
        ).astype(float)

        flags["computed_FSH_LH_ratio"] = (
            computed_ratio.reset_index(drop=True)
        )

        uncomputable_name = "flag_uncomputable_FSH_LH_ratio"

        flags[uncomputable_name] = (
            (~valid_ratio)
            .astype(bool)
            .reset_index(drop=True)
        )

        measurement_flags.append(uncomputable_name)

    # ---------------------------------------------------------
    # 9. Add flags to the original records
    # ---------------------------------------------------------
    flag_frame = pd.DataFrame(
        flags,
        index=result.index,
    )

    result = pd.concat(
        [result, flag_frame],
        axis=1,
    )

    # ---------------------------------------------------------
    # 10. Summarize quality categories
    # ---------------------------------------------------------
    result["has_extreme_value"] = (
        result[extreme_flags].any(axis=1)
        if extreme_flags
        else False
    )

    result["has_measurement_quality_issue"] = (
        result[measurement_flags].any(axis=1)
        if measurement_flags
        else False
    )

    result["has_record_integrity_issue"] = (
        result[record_flags].any(axis=1)
        if record_flags
        else False
    )

    result["has_source_format_issue"] = (
        result[source_flags].any(axis=1)
        if source_flags
        else False
    )

    result["has_quality_issue"] = result[
        [
            "has_measurement_quality_issue",
            "has_record_integrity_issue",
            "has_source_format_issue",
            "has_extreme_value",
        ]
    ].any(axis=1)

    return result


def show_flagged_patients(
    df: pd.DataFrame,
    preview_rows: int = 20,
) -> None:
    """Print a compact summary and preview of flagged records."""
    print("\n" + "=" * 70)
    print("CLINICAL QUALITY FLAG SUMMARY")
    print("=" * 70)

    print(f"Total records retained: {len(df)}")

    print(
        "Rows with any quality issue: "
        f"{int(df['has_quality_issue'].sum())}"
    )

    print(
        "Rows with measurement-quality issues: "
        f"{int(df['has_measurement_quality_issue'].sum())}"
    )

    print(
        "Rows with source-format issues: "
        f"{int(df['has_source_format_issue'].sum())}"
    )

    print(
        "Rows with patient-ID integrity issues: "
        f"{int(df['has_record_integrity_issue'].sum())}"
    )

    print(
        "Rows exceeding project screening thresholds: "
        f"{int(df['has_extreme_value'].sum())}"
    )

    boolean_flags = [
        column
        for column in df.columns
        if (
            column.startswith("flag_")
            and pd.api.types.is_bool_dtype(df[column])
        )
    ]

    counts = {
        column: int(df[column].sum())
        for column in boolean_flags
        if int(df[column].sum()) > 0
    }

    print("\nNonzero flag counts:")

    if counts:
        for name, count in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        ):
            print(f"{name:<45} {count:>5}")
    else:
        print("No flags raised.")

    priority = (
        df["has_measurement_quality_issue"]
        | df["has_record_integrity_issue"]
        | df["has_extreme_value"]
    )

    preview = df.loc[priority]

    display_columns = [
        column
        for column in [
            "Patient File No.",
            "PCOS (Y/N)",
            "BMI",
            "FSH(mIU/mL)",
            "LH(mIU/mL)",
            "AMH(ng/mL)",
            "computed_FSH_LH_ratio",
            "has_measurement_quality_issue",
            "has_extreme_value",
        ]
        if column in df.columns
    ]

    print(
        "\nPreview of records requiring measurement/ID review: "
        f"{len(preview)}"
    )

    if preview.empty:
        print("No measurement or ID review flags in this dataset.")
    else:
        print(
            preview[display_columns]
            .head(preview_rows)
            .to_string(index=False)
        )


def save_cleaned_dataset(
    df: pd.DataFrame,
    path: Path = OUTPUT_PATH,
) -> None:
    """Save the flagged dataset without dropping any patient rows."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(path, index=False)

    print("\n" + "=" * 70)
    print("FLAGGED DATASET SAVED")
    print("=" * 70)

    print(path.resolve())
    print(
        f"Rows retained: {len(df)}. "
        "No patients were deleted."
    )


def main() -> None:
    """Load, flag, summarize, and save the clinical dataset."""
    print("=" * 70)
    print("OVATWIN — CLINICAL DATA QUALITY FLAGGING")
    print("=" * 70)

    print(f"Loading dataset: {DATA_PATH}")

    raw = pd.read_csv(
        DATA_PATH,
        keep_default_na=False,
    )

    raw = _clean_columns(raw)

    loaded = load_clinical_data(DATA_PATH)
    validate_dataset(loaded)

    print(f"Original dataset shape: {loaded.shape}")

    flagged = create_quality_flags(
        loaded,
        raw_df=raw,
    )

    show_flagged_patients(flagged)
    save_cleaned_dataset(flagged)


if __name__ == "__main__":
    main()