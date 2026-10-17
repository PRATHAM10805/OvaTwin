"""Patient profiling and clinical observation quality checks."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

from src.HPO.patient_values import HPOPatientValues
from src.twin.parameter_mapping import (
    PatientTwinMapping,
    create_mapping,
)


# Candidate comparisons only; these mappings are not yet calibrated.
CANDIDATE_MODEL_OUTPUTS = {
    "LH": {
        "observation_group": "hormones",
        "model_species": "LH_bld",
    },
    "FSH": {
        "observation_group": "hormones",
        "model_species": "FSH_bld",
    },
    "Progesterone": {
        "observation_group": "hormones",
        "model_species": "P4",
    },
}


# Project review thresholds, not clinical diagnostic cutoffs.
SCREENING_THRESHOLDS = {
    "FSH": 50.0,
    "LH": 50.0,
    "AMH": 30.0,
    "Prolactin": 100.0,
    "Progesterone": 20.0,
    "TSH": 20.0,
}


def _usable_number(value: Any) -> bool:
    """Return True only for finite numeric observations."""
    if value is None or isinstance(value, bool):
        return False

    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _build_observation_groups(
    patient: HPOPatientValues,
    mapping: PatientTwinMapping,
) -> dict[str, dict[str, Any]]:
    """Build consistent observation groups without duplicate cycle fields."""

    hormones = dict(mapping.hormones)
    follicular = dict(mapping.follicular)
    uterine = dict(mapping.uterine)
    cycle = dict(mapping.cycle)
    anthropometric = dict(mapping.anthropometric)

    # Preserve the meaning of encoded cycle variables.
    # Do not treat encoded source values as literal cycle days.
    cycle.pop("length_encoded", None)

    cycle["cycle_status_encoded"] = patient.cycle_status_encoded
    cycle["cycle_length_encoded"] = patient.cycle_length_encoded

    # Add the derived anthropometric observation.
    anthropometric["waist_hip_ratio"] = patient.waist_hip_ratio

    return {
        "hormones": hormones,
        "follicular": follicular,
        "uterine": uterine,
        "cycle": cycle,
        "anthropometric": anthropometric,
    }


@dataclass
class PatientProfile:
    """Clinical profile for one OvaTwin patient."""

    patient: HPOPatientValues
    mapping: PatientTwinMapping
    data_quality_flags: list[str] = field(default_factory=list)

    def observation_groups(self) -> dict[str, dict[str, Any]]:
        """Return clinical observations grouped by category."""
        return _build_observation_groups(
            self.patient,
            self.mapping,
        )

    def observation_summary(self) -> dict:
        """Report observation completeness and data-quality flags."""

        groups = self.observation_groups()

        available_by_group = {}
        missing_by_group = {}

        total_count = 0
        available_count = 0

        for group_name, observations in groups.items():
            available = [
                name
                for name, value in observations.items()
                if _usable_number(value)
            ]

            missing = [
                name
                for name, value in observations.items()
                if not _usable_number(value)
            ]

            available_by_group[group_name] = available
            missing_by_group[group_name] = missing

            total_count += len(observations)
            available_count += len(available)

        missing_count = total_count - available_count

        completeness = (
            available_count / total_count
            if total_count
            else 0.0
        )

        return {
            "patient_id": self.patient.patient_id,
            "available_observation_count": available_count,
            "total_observation_count": total_count,
            "missing_observation_count": missing_count,
            "completeness_fraction": completeness,
            "available_by_group": available_by_group,
            "missing_by_group": missing_by_group,
            "data_quality_flags": list(self.data_quality_flags),
        }

    def comparison_readiness(self) -> dict:
        """
        Report candidate comparisons between observations and model outputs.

        Matching names do not guarantee matching units, sampling times,
        or clinical meaning. These results do not establish calibration.
        """
        observations = self.mapping.hormones
        results = {}

        for observation_name, config in CANDIDATE_MODEL_OUTPUTS.items():
            value = observations.get(observation_name)

            if not _usable_number(value):
                status = "MISSING_OBSERVATION"
            else:
                status = "NEEDS_UNIT_AND_TIMING_ALIGNMENT"

            results[observation_name] = {
                "observed_value": (
                    float(value)
                    if _usable_number(value)
                    else None
                ),
                "candidate_model_species": config["model_species"],
                "status": status,
            }

        # These observations have no verified direct output mappings
        # in the current OvaTwin calibration interface.
        for observation_name in ("AMH", "Prolactin", "TSH"):
            value = observations.get(observation_name)

            results[observation_name] = {
                "observed_value": (
                    float(value)
                    if _usable_number(value)
                    else None
                ),
                "candidate_model_species": None,
                "status": "NO_VERIFIED_DIRECT_OUTPUT_MAPPING",
            }

        return results

    def summary(self) -> dict:
        """Return observation and comparison-readiness summaries."""
        return {
            **self.observation_summary(),
            "comparison_readiness": self.comparison_readiness(),
        }


def validate_patient_observations(
    patient: HPOPatientValues,
) -> list[str]:
    """Flag invalid and suspicious observations without modifying them."""

    mapping = create_mapping(patient)
    groups = _build_observation_groups(patient, mapping)

    flags = []

    for group_name, observations in groups.items():
        for observation_name, value in observations.items():

            if value is None:
                continue

            if not _usable_number(value):
                flags.append(
                    f"{group_name}.{observation_name}: "
                    "value is not a finite number"
                )
                continue

            if float(value) < 0:
                flags.append(
                    f"{group_name}.{observation_name}: "
                    "negative value requires review"
                )

    # Flag measurements that exceed project review thresholds.
    # These flags are not clinical diagnoses.
    hormone_values = {
        "FSH": patient.fsh,
        "LH": patient.lh,
        "AMH": patient.amh,
        "Prolactin": patient.prolactin,
        "Progesterone": patient.progesterone,
        "TSH": patient.tsh,
    }

    for name, threshold in SCREENING_THRESHOLDS.items():
        value = hormone_values[name]

        if _usable_number(value) and float(value) > threshold:
            flags.append(
                f"hormones.{name}: exceeds project review threshold "
                f"> {threshold:g}; review the source measurement"
            )

    if (
        _usable_number(patient.fsh_lh_ratio)
        and float(patient.fsh_lh_ratio) > 100
    ):
        flags.append(
            "hormones.FSH_LH_ratio: exceeds project review threshold "
            "> 100; review the original FSH and LH measurements"
        )

    return flags


def create_patient_profile(
    patient: HPOPatientValues,
) -> PatientProfile:
    """Create a profile without changing clinical observations or HPO parameters."""

    mapping = create_mapping(patient)

    return PatientProfile(
        patient=patient,
        mapping=mapping,
        data_quality_flags=validate_patient_observations(patient),
    )