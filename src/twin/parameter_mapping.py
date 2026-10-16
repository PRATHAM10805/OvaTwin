from dataclasses import dataclass, field
from typing import Any, List

from src.HPO.patient_values import HPOPatientValues
from src.twin.roblitz_mapping import (
    map_p1_b_syn_LH,
    map_p17_b_rel_FSH,
    MappingAuditEntry,
)


@dataclass
class PatientTwinMapping:
    """
    Patient-specific observations and personalized ODE parameters used by the digital twin.

    Observations: static clinical observations from the patient dataset.
    ODE Parameters: phenotype-informed personalized kinetic parameter overrides.
    """

    patient_id: int

    hormones: dict[str, float | None] = field(default_factory=dict)
    follicular: dict[str, float | None] = field(default_factory=dict)
    uterine: dict[str, float | None] = field(default_factory=dict)
    cycle: dict[str, float | None] = field(default_factory=dict)
    anthropometric: dict[str, float | None] = field(default_factory=dict)

    # Phase 7: Phenotype-informed ODE parameter overrides & audit log
    ode_parameters: dict[str, float] = field(default_factory=dict)
    mapping_audit: List[MappingAuditEntry] = field(default_factory=list)


def create_mapping(
    patient: HPOPatientValues,
    personalize_ode: bool = True,
) -> PatientTwinMapping:
    """
    Convert clinical/HPO observations into the patient-specific
    representation used by the digital twin.

    If personalize_ode is True, maps relevant clinical observations
    to validated Röblitz kinetic parameters (p1 and p17).
    """

    ode_params: dict[str, float] = {}
    audit_log: List[MappingAuditEntry] = []

    if personalize_ode:
        # Pituitary LH synthesis rate (p1)
        p1_val, p1_audit = map_p1_b_syn_LH(patient)
        ode_params["p1"] = p1_val
        ode_params["b_syn_LH"] = p1_val
        audit_log.append(p1_audit)

        # Pituitary FSH release rate (p17)
        p17_val, p17_audit = map_p17_b_rel_FSH(patient)
        ode_params["p17"] = p17_val
        ode_params["b_rel_FSH"] = p17_val
        audit_log.append(p17_audit)

    return PatientTwinMapping(
        patient_id=patient.patient_id,
        ode_parameters=ode_params,
        mapping_audit=audit_log,

        hormones={
            "FSH": patient.fsh,
            "LH": patient.lh,
            "AMH": patient.amh,
            "Progesterone": patient.progesterone,
            "Prolactin": patient.prolactin,
            "TSH": patient.tsh,
            "FSH_LH_ratio": patient.fsh_lh_ratio,
        },

        follicular={
            "left_count": patient.follicle_count_left,
            "right_count": patient.follicle_count_right,
            "total_count": patient.total_follicle_count,
            "left_size": patient.follicle_size_left,
            "right_size": patient.follicle_size_right,
            "mean_size": patient.mean_follicle_size,
        },

        uterine={
            "endometrium": patient.endometrium,
        },

        cycle={
            "regular": patient.cycle_regular,
            "length_encoded": patient.cycle_length_encoded,
        },

        anthropometric={
            "age": patient.age,
            "weight": patient.weight,
            "height": patient.height,
            "bmi": patient.bmi,
        },
    )