"""Read-only audit of missing, malformed, non-finite, and suspicious clinical data."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.Clinical.clinical_loader import (
    load_clinical_data,
    validate_dataset,
)
from src.Clinical.clinical_cleaning import (
    DATA_PATH,
    EXTREME_THRESHOLDS,
    NUMERIC_MEASUREMENT_COLUMNS,
    _clean_columns,
    _missing_mask,
    _numeric,
    create_quality_flags,
)


def section(title: str) -> None:
    """Print a readable report section."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def report_overview(df: pd.DataFrame) -> None:
    """Report dataset dimensions and target distribution."""
    section("DATASET OVERVIEW")

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    if "PCOS (Y/N)" in df.columns:
        labels = df["PCOS (Y/N)"].value_counts(
            dropna=False
        ).sort_index()

        print("\nPCOS label distribution:")
        print(labels.to_string())

        print("\nLabel proportions:")
        proportions = (
            df["PCOS (Y/N)"]
            .value_counts(normalize=True, dropna=False)
            .sort_index()
            * 100
        )
        print(
            proportions.round(2)
            .astype(str)
            .add("%")
            .to_string()
        )


def report_missing(raw: pd.DataFrame) -> None:
    """Count true missing values separately from malformed text."""
    section("ACTUAL MISSING VALUES")

    counts = raw.apply(
        lambda column: int(_missing_mask(column).sum())
    )

    nonzero = counts[counts > 0].sort_values(
        ascending=False
    )

    if nonzero.empty:
        print("No actual missing cells detected.")
    else:
        summary = pd.DataFrame(
            {
                "missing_count": nonzero,
                "missing_percent": (
                    nonzero / len(raw) * 100
                ).round(2),
            }
        )

        print(summary.to_string())

    print(
        f"\nTotal actual missing cells: "
        f"{int(counts.sum())} / {raw.size}"
    )


def report_numeric_source_issues(
    raw: pd.DataFrame,
) -> None:
    """Identify nonempty text that cannot be parsed as numeric."""
    section("MALFORMED VALUES IN PRIMARY NUMERIC MEASUREMENTS")

    rows = []

    for column in NUMERIC_MEASUREMENT_COLUMNS:
        if column not in raw.columns:
            continue

        source = raw[column]
        missing = _missing_mask(source)
        values = _numeric(source)

        malformed = (~missing) & values.isna()
        count = int(malformed.sum())

        if count:
            examples = (
                source.loc[malformed]
                .astype(str)
                .value_counts()
                .head(3)
                .to_dict()
            )
            rows.append((column, count, examples))

    if not rows:
        print(
            "No malformed values detected in "
            "inspected numeric fields."
        )
    else:
        for column, count, examples in rows:
            print(
                f"{column}: {count} malformed nonempty "
                f"value(s); examples={examples}"
            )

    print(
        "These are source-format/parsing issues, "
        "not actual missing values."
    )

    section("SOURCE-DERIVED FIELD QUALITY")

    derived = [
        column
        for column in ["BMI", "Waist:Hip Ratio", "FSH/LH"]
        if column in raw.columns
    ]

    if not derived:
        print("No source-derived fields found.")

    for column in derived:
        source = raw[column]
        missing = _missing_mask(source)
        values = _numeric(source)
        malformed = (~missing) & values.isna()

        print(
            f"{column}: missing={int(missing.sum())}, "
            f"malformed/non-numeric={int(malformed.sum())}"
        )

    print(
        "BMI and waist/hip ratio can be recalculated "
        "from weight/height and waist/hip inputs "
        "when those inputs are valid."
    )
    print(
        "The source FSH/LH ratio should be recalculated "
        "from the FSH and LH columns."
    )


def report_patient_ids(raw: pd.DataFrame) -> None:
    """Check missing, malformed, non-integer, and duplicate IDs."""
    section("PATIENT ID INTEGRITY")

    column = "Patient File No."

    if column not in raw.columns:
        print(f"Required ID column not found: {column}")
        return

    source = raw[column]
    missing = _missing_mask(source)
    ids = _numeric(source)

    finite = (
        ids.notna()
        & np.isfinite(ids.astype(float))
    )

    invalid = (~missing) & (
        ids.isna()
        | (~finite)
        | (finite & ids.mod(1).ne(0))
    )

    valid = (~missing) & (~invalid)
    valid_ids = ids.where(valid)

    duplicates = (
        valid_ids.notna()
        & valid_ids.duplicated(keep=False)
    )

    print(f"Missing IDs: {int(missing.sum())}")
    print(f"Invalid/non-integer IDs: {int(invalid.sum())}")
    print(
        "Rows belonging to duplicated valid IDs: "
        f"{int(duplicates.sum())}"
    )
    print(
        "Unique valid patient IDs: "
        f"{int(ids.where(valid).nunique())}"
    )


def report_nonfinite(raw: pd.DataFrame) -> None:
    """Report actual positive or negative infinity separately."""
    section("TRUE NON-FINITE NUMERIC VALUES")

    found = False

    columns = list(
        dict.fromkeys(
            NUMERIC_MEASUREMENT_COLUMNS
            + ["BMI", "Waist:Hip Ratio", "FSH/LH"]
            + list(EXTREME_THRESHOLDS)
        )
    )

    for column in columns:
        if column not in raw.columns:
            continue

        values = _numeric(raw[column])

        mask = (
            values.notna()
            & ~np.isfinite(values.astype(float))
        )

        count = int(mask.sum())

        if count:
            found = True
            print(f"{column}: {count}")

    if not found:
        print("No true +inf or -inf values detected.")

    print(
        "NaN/missing values and malformed text are "
        "reported separately above."
    )


def report_thresholds(raw: pd.DataFrame) -> None:
    """Report values exceeding the project's review thresholds."""
    section("SCREENING-THRESHOLD REVIEW")

    print(
        "These are existing project review thresholds, "
        "not validated clinical diagnostic cutoffs."
    )

    id_columns = [
        column
        for column in ["Patient File No.", "PCOS (Y/N)"]
        if column in raw.columns
    ]

    context_columns = [
        column
        for column in [
            "Age (yrs)",
            "Cycle(R/I)",
            "Cycle length(days)",
        ]
        if column in raw.columns
    ]

    for column, threshold in EXTREME_THRESHOLDS.items():
        if column not in raw.columns:
            continue

        values = _numeric(raw[column])

        mask = (
            values.notna()
            & np.isfinite(values.astype(float))
            & values.gt(threshold)
        )

        selected_columns = (
            id_columns + [column] + context_columns
        )

        rows = raw.loc[mask, selected_columns]

        print(
            f"\n{column}: review threshold > {threshold}; "
            f"flagged values={len(rows)}"
        )

        if not rows.empty:
            print(rows.to_string(index=False))


def report_negative_and_zero(
    raw: pd.DataFrame,
) -> None:
    """Inspect negative values and potentially relevant zeros."""
    section("NEGATIVE NUMERIC MEASUREMENTS")

    found = False

    for column in NUMERIC_MEASUREMENT_COLUMNS:
        if column not in raw.columns:
            continue

        values = _numeric(raw[column])

        mask = (
            values.notna()
            & np.isfinite(values.astype(float))
            & values.lt(0)
        )

        if mask.any():
            found = True
            print(
                f"{column}: {int(mask.sum())} negative value(s)"
            )

    if not found:
        print("No negative values detected in inspected measurements.")

    section("ZERO-VALUED MEASUREMENTS")

    zero_columns = [
        column
        for column in [
            "FSH(mIU/mL)",
            "LH(mIU/mL)",
            "AMH(ng/mL)",
            "PRL(ng/mL)",
            "PRG(ng/mL)",
            "TSH (mIU/L)",
            "Follicle No. (L)",
            "Follicle No. (R)",
            "Avg. F size (L) (mm)",
            "Avg. F size (R) (mm)",
            "Endometrium (mm)",
        ]
        if column in raw.columns
    ]

    for column in zero_columns:
        values = _numeric(raw[column])

        zero_count = int(
            (
                values.notna()
                & np.isfinite(values.astype(float))
                & values.eq(0)
            ).sum()
        )

        print(f"{column:<30} zeros = {zero_count}")


def report_ratio_and_cycle(
    raw: pd.DataFrame,
) -> None:
    """Review the original FSH/LH ratio and cycle-variable encodings."""
    section("ORIGINAL FSH/LH RATIO COLUMN")

    if "FSH/LH" in raw.columns:
        ratio = raw["FSH/LH"]
        missing = _missing_mask(ratio)
        values = _numeric(ratio)
        malformed = (~missing) & values.isna()

        print(f"Rows in source ratio column: {len(ratio)}")
        print(f"Missing source ratios: {int(missing.sum())}")
        print(
            "Nonnumeric nonempty source ratios: "
            f"{int(malformed.sum())}"
        )

        if malformed.any():
            print("\nExamples of nonnumeric source values:")
            print(
                ratio.loc[malformed]
                .astype(str)
                .value_counts()
                .head(10)
                .to_string()
            )

    if {
        "FSH(mIU/mL)",
        "LH(mIU/mL)",
    }.issubset(raw.columns):

        fsh = _numeric(raw["FSH(mIU/mL)"])
        lh = _numeric(raw["LH(mIU/mL)"])

        valid = (
            fsh.notna()
            & lh.notna()
            & np.isfinite(fsh.astype(float))
            & np.isfinite(lh.astype(float))
            & lh.gt(0)
            & fsh.ge(0)
        )

        print(
            "Rows where a ratio can be computed from FSH "
            f"and LH: {int(valid.sum())}"
        )

    section("CYCLE VARIABLE ENCODING")

    for column in ["Cycle(R/I)", "Cycle length(days)"]:
        if column in raw.columns:
            print(f"\n{column} value counts:")
            print(
                raw[column]
                .value_counts(dropna=False)
                .sort_index()
                .to_string()
            )

    if {
        "Cycle(R/I)",
        "Cycle length(days)",
    }.issubset(raw.columns):
        print("\nCycle regularity vs encoded cycle-length values:")

        print(
            pd.crosstab(
                raw["Cycle(R/I)"],
                raw["Cycle length(days)"],
                dropna=False,
            ).to_string()
        )

    print(
        "\nInterpretation warning: numeric encoding alone "
        "does not establish literal cycle lengths in days."
    )


def report_flag_summary(
    loaded: pd.DataFrame,
    raw: pd.DataFrame,
) -> None:
    """Summarize distinct quality-issue categories."""
    section("QUALITY FLAG SUMMARY")

    flagged = create_quality_flags(loaded, raw_df=raw)

    print(f"Total rows: {len(flagged)}")
    print(
        "Rows with any quality issue: "
        f"{int(flagged['has_quality_issue'].sum())}"
    )
    print(
        "Rows with measurement-quality issues: "
        f"{int(flagged['has_measurement_quality_issue'].sum())}"
    )
    print(
        "Rows with source-format issues: "
        f"{int(flagged['has_source_format_issue'].sum())}"
    )
    print(
        "Rows with patient-ID integrity issues: "
        f"{int(flagged['has_record_integrity_issue'].sum())}"
    )
    print(
        "Rows exceeding project screening thresholds: "
        f"{int(flagged['has_extreme_value'].sum())}"
    )

    print(
        "\nThis audit is read-only: it does not save data "
        "or delete rows."
    )


def main() -> None:
    """Run the complete read-only clinical audit."""
    print("=" * 70)
    print("OVATWIN — CLINICAL DATA QUALITY AUDIT")
    print("=" * 70)
    print(f"Dataset: {DATA_PATH}")

    raw = pd.read_csv(
        DATA_PATH,
        keep_default_na=False,
    )
    raw = _clean_columns(raw)

    loaded = load_clinical_data(DATA_PATH)
    validate_dataset(loaded)

    report_overview(loaded)
    report_missing(raw)
    report_numeric_source_issues(raw)
    report_patient_ids(raw)
    report_nonfinite(raw)
    report_thresholds(raw)
    report_negative_and_zero(raw)
    report_ratio_and_cycle(raw)
    report_flag_summary(loaded, raw)


if __name__ == "__main__":
    main()