"""Build standardized patient observations from a clinical CSV row."""

from __future__ import annotations

import math
from typing import Any, Iterable

from src.HPO.patient_values import HPOPatientValues


def _to_float(value: Any) -> float | None:
    """Return a finite float, or None for missing/malformed/non-finite data."""
    if value is None or isinstance(value, bool):
        return None

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None

    return number if math.isfinite(number) else None


def _first_number(row, keys: Iterable[str]) -> float | None:
    """Return the first usable value from preferred columns in order."""
    for key in keys:
        try:
            value = _to_float(row.get(key))
        except AttributeError:
            value = None

        if value is not None:
            return value

    return None


def _patient_id(row) -> int:
    """Validate Patient File No. before converting it to an integer."""
    try:
        value = _to_float(row.get("Patient File No."))
    except AttributeError as exc:
        raise ValueError(
            "Expected a row-like object with Patient File No."
        ) from exc

    if value is None or value <= 0 or not value.is_integer():
        raise ValueError(
            "Patient File No. must be a finite positive integer; "
            f"received {row.get('Patient File No.')!r}."
        )

    return int(value)


def build_hpo_patient_values(row) -> HPOPatientValues:
    """Convert one clinical row to observations, not model parameters.

    Supports either the raw cohort or the OvaTwin preprocessed dataset.
    Recalculated columns are preferred when available, with safe fallbacks
    to calculations from the underlying raw measurements.
    """
    fsh = _first_number(row, ["FSH(mIU/mL)"])
    lh = _first_number(row, ["LH(mIU/mL)"])

    fsh_lh_ratio = _first_number(
        row,
        ["FSH_LH_ratio", "computed_FSH_LH_ratio"],
    )

    if (
        fsh_lh_ratio is None
        and fsh is not None
        and lh is not None
        and lh > 0
        and fsh >= 0
    ):
        fsh_lh_ratio = fsh / lh

    left_follicles = _first_number(row, ["Follicle No. (L)"])
    right_follicles = _first_number(row, ["Follicle No. (R)"])

    total_follicles = _first_number(
        row,
        ["total_follicle_count"],
    )

    if (
        total_follicles is None
        and left_follicles is not None
        and right_follicles is not None
        and left_follicles >= 0
        and right_follicles >= 0
    ):
        total_follicles = left_follicles + right_follicles

    left_size = _first_number(row, ["Avg. F size (L) (mm)"])
    right_size = _first_number(row, ["Avg. F size (R) (mm)"])

    mean_size = _first_number(
        row,
        ["mean_follicle_size_mm"],
    )

    if (
        mean_size is None
        and left_size is not None
        and right_size is not None
        and left_size >= 0
        and right_size >= 0
    ):
        mean_size = (left_size + right_size) / 2.0

    weight = _first_number(row, ["Weight (Kg)"])
    height = _first_number(row, ["Height(Cm)"])

    bmi = _first_number(
        row,
        ["computed_BMI", "BMI"],
    )

    if bmi is None and weight is not None and height is not None:
        if weight > 0 and height > 0:
            bmi = weight / ((height / 100.0) ** 2)

    waist_hip_ratio = _first_number(
        row,
        [
            "computed_Waist_Hip_Ratio",
            "computed_waist_hip_ratio",
            "Waist:Hip Ratio",
        ],
    )

    if waist_hip_ratio is None:
        waist = _first_number(row, ["Waist(inch)"])
        hip = _first_number(row, ["Hip(inch)"])

        if (
            waist is not None
            and hip is not None
            and waist >= 0
            and hip > 0
        ):
            waist_hip_ratio = waist / hip

    return HPOPatientValues(
        patient_id=_patient_id(row),

        # Demographics
        age=_first_number(row, ["Age (yrs)"]),
        weight=weight,
        height=height,
        bmi=bmi,
        waist_hip_ratio=waist_hip_ratio,

        # Hormones
        fsh=fsh,
        lh=lh,
        amh=_first_number(row, ["AMH(ng/mL)"]),
        progesterone=_first_number(row, ["PRG(ng/mL)"]),
        prolactin=_first_number(row, ["PRL(ng/mL)"]),
        tsh=_first_number(row, ["TSH (mIU/L)"]),
        fsh_lh_ratio=fsh_lh_ratio,

        # Follicles
        follicle_count_left=left_follicles,
        follicle_count_right=right_follicles,
        total_follicle_count=total_follicles,
        follicle_size_left=left_size,
        follicle_size_right=right_size,
        mean_follicle_size=mean_size,

        # Uterine measurement
        endometrium=_first_number(row, ["Endometrium (mm)"]),

        # Cycle(R/I) values are encoded category codes in this dataset.
        # Do not populate cycle_regular until the code meanings are verified.
        cycle_regular=None,
        cycle_status_encoded=_first_number(row, ["Cycle(R/I)"]),
        cycle_length_encoded=_first_number(row, ["Cycle length(days)"]),
    )