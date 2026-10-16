from dataclasses import dataclass, field
from typing import Any, List

from src.HPO.patient_values import HPOPatientValues
from src.twin.roblitz_mapping import (
    map_p1_b_syn_LH,
    map_p17_b_rel_FSH,
    map_p49_k_AF1,
    map_p51_k_AF2_AF3,
    map_p173_b_IhB,
    map_core_5_parameters,
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
    to all 5 validated Core Röblitz kinetic parameters:
    (p1, p17, p49, p51, p173).
    """

    ode_params: dict[str, float] = {}
    audit_log: List[MappingAuditEntry] = []

    if personalize_ode:
        ode_params, audit_log = map_core_5_parameters(patient)

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