"""
roblitz_mapping.py

Generic two-layer bounded parameter mapping engine for the Röblitz 2013 HPO model.
Implements Phase 7.4:
  - Layer 1: Soft bounded non-linear scaling (hyperbolic tangent, tanh).
  - Layer 2: Hard physiological safety clamping via AuthoritativeParameterRegistry.
  - Transparent audit logging and explicit missing-value handling.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math

from src.HPO.parameters import (
    AuthoritativeParameterRegistry,
    get_authoritative_registry,
)

ROOT_DIR = Path(__file__).resolve().parents[2]


# =====================================================================
# Clinical Reference Population Medians (x_ref)
# =====================================================================
# Reference values representing standard physiological medians for young adult females
# during the early-to-mid follicular phase.
CLINICAL_REFERENCES = {
    "LH": 5.0,                   # mIU/mL
    "FSH": 5.5,                  # mIU/mL
    "FSH_LH_RATIO": 1.2,         # FSH / LH healthy ratio
    "TOTAL_FOLLICLE_COUNT": 12.0,# Combined antral follicle count (left + right)
    "AMH": 2.5,                  # ng/mL
    "MEAN_FOLLICLE_SIZE": 14.0,  # mm
    "PROGESTERONE": 0.8,         # ng/mL
    "ESTRADIOL": 50.0,           # pg/mL
    "CYCLE_REGULAR": 1.0,        # 1.0 = regular, 0.0 = irregular
    "BMI": 22.0,                 # kg/m^2
}


@dataclass(frozen=True)
class MappingAuditEntry:
    """Audit log entry documenting every parameter mapping decision."""
    parameter_id: str
    parameter_name: str
    clinical_marker: str
    observed_value: Optional[float]
    reference_value: float
    scale_factor: float
    baseline_value: float
    final_value: float
    status: str  # 'personalized', 'clamped', 'missing_fallback'
    reason: str = ""

    def __repr__(self) -> str:
        obs_str = f"{self.observed_value:.2f}" if self.observed_value is not None else "None"
        return (
            f"Audit({self.parameter_id} | {self.parameter_name} | {self.clinical_marker}={obs_str} "
            f"-> scale={self.scale_factor:.3f} -> final={self.final_value:.4f} [{self.status}])"
        )


def bounded_scale(
    value: Optional[float],
    reference: float,
    max_delta: float = 0.35,
    alpha: float = 0.8,
    direction: int = 1,
) -> float:
    """
    Layer 1: Soft bounded non-linear scaling using hyperbolic tangent (tanh).

    Formula:
        S(x) = 1.0 + max_delta * tanh(direction * alpha * (x - reference) / reference)

    Properties:
        - When value == reference: S = 1.0 (exact baseline preserved).
        - Value strictly bounded in [1.0 - max_delta, 1.0 + max_delta].
        - If value is None, NaN, or Inf: returns 1.0 (graceful fallback).
        - If direction == -1: inverse correlation (higher clinical marker reduces parameter).
    """
    if reference <= 0:
        raise ValueError(f"Reference value must be strictly positive, got {reference}")

    if value is None or math.isnan(value) or math.isinf(value):
        return 1.0

    if value < 0:
        # Negative clinical marker is non-physical; treat as zero
        value = 0.0

    normalized_diff = (value - reference) / reference
    z = float(direction) * alpha * normalized_diff
    scale = 1.0 + max_delta * math.tanh(z)

    return scale


def map_parameter_value(
    param_key: str,
    clinical_value: Optional[float],
    reference_value: float,
    marker_name: str,
    max_delta: float = 0.35,
    alpha: float = 0.8,
    direction: int = 1,
    registry: Optional[AuthoritativeParameterRegistry] = None,
) -> Tuple[float, MappingAuditEntry]:
    """
    Maps a single clinical observation to its corresponding Röblitz model parameter.
    Combines Layer 1 (soft tanh scale) and Layer 2 (hard safety clamp).
    """
    reg = registry or get_authoritative_registry()
    meta = reg.get(param_key)
    base_val = meta.baseline_value

    # Check for missing clinical observation
    if clinical_value is None or math.isnan(clinical_value) or math.isinf(clinical_value):
        scale = 1.0
        final_val = base_val
        status = "missing_fallback"
        reason = (
            f"Clinical observation '{marker_name}' is missing; "
            f"assigned published baseline ({base_val:.4f})."
        )
    else:
        # Layer 1: Soft bounded scaling
        scale = bounded_scale(
            value=clinical_value,
            reference=reference_value,
            max_delta=max_delta,
            alpha=alpha,
            direction=direction,
        )
        proposed_val = base_val * scale

        # Layer 2: Hard physiological safety clamp
        final_val = meta.clamp(proposed_val)

        if final_val != proposed_val:
            status = "clamped"
            reason = (
                f"Proposed value {proposed_val:.4f} exceeded bounds "
                f"[{meta.min_bound:.4f}, {meta.max_bound:.4f}]; clamped to {final_val:.4f}."
            )
        else:
            status = "personalized"
            reason = (
                f"Scaled by {scale:.3f} from baseline {base_val:.4f} "
                f"based on '{marker_name}'={clinical_value:.2f}."
            )

    audit_entry = MappingAuditEntry(
        parameter_id=meta.id,
        parameter_name=meta.name,
        clinical_marker=marker_name,
        observed_value=clinical_value if clinical_value is not None and not math.isnan(clinical_value) else None,
        reference_value=reference_value,
        scale_factor=scale,
        baseline_value=base_val,
        final_value=final_val,
        status=status,
        reason=reason,
    )

    return (final_val, audit_entry)


def map_p1_b_syn_LH(
    patient: "HPOPatientValues",
    registry: Optional[AuthoritativeParameterRegistry] = None,
) -> Tuple[float, MappingAuditEntry]:
    """
    Phase 7.5: Maps patient clinical LH observations to Röblitz parameter p1 (b_syn_LH).

    Dual-Signal Gonadotropin Logic:
      1. Primary Signal (Serum LH):
         Drives pituitary LH basal synthesis.
         Reference median: 5.0 mIU/mL. Max perturbation contribution: +/-30%.
      2. Secondary Signal (Computed FSH/LH ratio):
         Inverted gonadotropin balance modulation. Lower FSH/LH indicates higher LH dominance.
         Reference median: 1.2. Max perturbation contribution: +/-10% (direction = -1).

    Layer 1 (Soft Bounds): Combined perturbation strictly bounded to [0.65, 1.35].
    Layer 2 (Hard Safety Clamp): Enforced via AuthoritativeParameterRegistry [2923.97, 18274.79].
    """
    reg = registry or get_authoritative_registry()
    meta_p1 = reg.get("p1")
    base_p1 = meta_p1.baseline_value

    lh_val = patient.lh
    ratio_val = patient.fsh_lh_ratio

    # If primary LH is completely missing, fall back to baseline
    if lh_val is None or math.isnan(lh_val) or math.isinf(lh_val):
        final_val = base_p1
        audit = MappingAuditEntry(
            parameter_id="p1",
            parameter_name="b_syn_LH",
            clinical_marker="LH",
            observed_value=None,
            reference_value=CLINICAL_REFERENCES["LH"],
            scale_factor=1.0,
            baseline_value=base_p1,
            final_value=final_val,
            status="missing_fallback",
            reason="Serum LH observation is missing; assigned published baseline.",
        )
        return (final_val, audit)

    # 1. Primary scale from serum LH (max_delta=0.30, alpha=0.7)
    s_lh = bounded_scale(
        value=lh_val,
        reference=CLINICAL_REFERENCES["LH"],
        max_delta=0.30,
        alpha=0.7,
        direction=1,
    )
    delta_lh = s_lh - 1.0

    # 2. Secondary scale from FSH/LH ratio (max_delta=0.10, alpha=0.6, direction=-1)
    if ratio_val is not None and not math.isnan(ratio_val) and not math.isinf(ratio_val):
        s_ratio = bounded_scale(
            value=ratio_val,
            reference=CLINICAL_REFERENCES["FSH_LH_RATIO"],
            max_delta=0.10,
            alpha=0.6,
            direction=-1,
        )
        delta_ratio = s_ratio - 1.0
    else:
        delta_ratio = 0.0

    # Combined perturbation with soft bounding clamp [-0.35, +0.35]
    total_delta = max(-0.35, min(delta_lh + delta_ratio, 0.35))
    scale_net = 1.0 + total_delta

    proposed_val = base_p1 * scale_net

    # Layer 2: Hard physiological safety clamp
    final_val = meta_p1.clamp(proposed_val)

    status = "clamped" if final_val != proposed_val else "personalized"
    ratio_str = f"{ratio_val:.2f}" if ratio_val is not None else "N/A"
    reason = (
        f"Personalized via LH={lh_val:.2f} (delta={delta_lh:+.3f}) and "
        f"FSH/LH={ratio_str} (delta={delta_ratio:+.3f}); "
        f"net scale={scale_net:.3f}."
    )

    audit = MappingAuditEntry(
        parameter_id="p1",
        parameter_name="b_syn_LH",
        clinical_marker="LH+FSH_LH_ratio",
        observed_value=lh_val,
        reference_value=CLINICAL_REFERENCES["LH"],
        scale_factor=scale_net,
        baseline_value=base_p1,
        final_value=final_val,
        status=status,
        reason=reason,
    )

    return (final_val, audit)


def map_p17_b_rel_FSH(
    patient: "HPOPatientValues",
    registry: Optional[AuthoritativeParameterRegistry] = None,
) -> Tuple[float, MappingAuditEntry]:
    """
    Phase 7.8: Maps patient clinical FSH observations to Röblitz parameter p17 (b_rel_FSH).

    Dual-Signal Gonadotropin Logic:
      1. Primary Signal (Serum FSH):
         Governs basal release rate of stored pituitary FSH into systemic blood circulation.
         Reference median: 5.5 mIU/mL. Max perturbation contribution: +/-30%.
      2. Secondary Signal (Computed FSH/LH ratio):
         Reflects relative gonadotropin balance. Higher FSH/LH supports relative FSH release.
         Reference median: 1.2. Max perturbation contribution: +/-8% (direction = +1).

    Layer 1 (Soft Bounds): Combined perturbation strictly bounded to [0.65, 1.35].
    Layer 2 (Hard Safety Clamp): Enforced via AuthoritativeParameterRegistry [0.0228, 0.1425].
    """
    reg = registry or get_authoritative_registry()
    meta_p17 = reg.get("p17")
    base_p17 = meta_p17.baseline_value

    fsh_val = patient.fsh
    ratio_val = patient.fsh_lh_ratio

    # If primary FSH is completely missing, fall back to baseline
    if fsh_val is None or math.isnan(fsh_val) or math.isinf(fsh_val):
        final_val = base_p17
        audit = MappingAuditEntry(
            parameter_id="p17",
            parameter_name="b_rel_FSH",
            clinical_marker="FSH",
            observed_value=None,
            reference_value=CLINICAL_REFERENCES["FSH"],
            scale_factor=1.0,
            baseline_value=base_p17,
            final_value=final_val,
            status="missing_fallback",
            reason="Serum FSH observation is missing; assigned published baseline.",
        )
        return (final_val, audit)

    # 1. Primary scale from serum FSH (max_delta=0.30, alpha=0.7)
    s_fsh = bounded_scale(
        value=fsh_val,
        reference=CLINICAL_REFERENCES["FSH"],
        max_delta=0.30,
        alpha=0.7,
        direction=1,
    )
    delta_fsh = s_fsh - 1.0

    # 2. Secondary scale from FSH/LH ratio (max_delta=0.08, alpha=0.5, direction=+1)
    if ratio_val is not None and not math.isnan(ratio_val) and not math.isinf(ratio_val):
        s_ratio = bounded_scale(
            value=ratio_val,
            reference=CLINICAL_REFERENCES["FSH_LH_RATIO"],
            max_delta=0.08,
            alpha=0.5,
            direction=1,
        )
        delta_ratio = s_ratio - 1.0
    else:
        delta_ratio = 0.0

    # Combined perturbation with soft bounding clamp [-0.35, +0.35]
    total_delta = max(-0.35, min(delta_fsh + delta_ratio, 0.35))
    scale_net = 1.0 + total_delta

    proposed_val = base_p17 * scale_net

    # Layer 2: Hard physiological safety clamp
    final_val = meta_p17.clamp(proposed_val)

    status = "clamped" if final_val != proposed_val else "personalized"
    ratio_str = f"{ratio_val:.2f}" if ratio_val is not None else "N/A"
    reason = (
        f"Personalized via FSH={fsh_val:.2f} (delta={delta_fsh:+.3f}) and "
        f"FSH/LH={ratio_str} (delta={delta_ratio:+.3f}); "
        f"net scale={scale_net:.3f}."
    )

    audit = MappingAuditEntry(
        parameter_id="p17",
        parameter_name="b_rel_FSH",
        clinical_marker="FSH+FSH_LH_ratio",
        observed_value=fsh_val,
        reference_value=CLINICAL_REFERENCES["FSH"],
        scale_factor=scale_net,
        baseline_value=base_p17,
        final_value=final_val,
        status=status,
        reason=reason,
    )

    return (final_val, audit)


def map_p49_k_AF1(
    patient: "HPOPatientValues",
    registry: Optional[AuthoritativeParameterRegistry] = None,
) -> Tuple[float, MappingAuditEntry]:
    """
    Phase 7.9: Maps patient antral follicle count to Röblitz parameter p49 (k_AF1).

    Biological Mechanism:
      Governs early antral follicle recruitment rate (re43 -> AF1).
      In PCOS, the expanded preantral follicle pool generates excessive recruitment into AF1.
      Primary Signal: total_follicle_count (left + right).
      Reference Median: 12.0 follicles. Max perturbation: +/-30%.

    Layer 1 (Soft Bounds): Bounded strictly to [0.65, 1.35].
    Layer 2 (Hard Safety Clamp): Enforced via AuthoritativeParameterRegistry [1.4647, 9.1545].
    """
    reg = registry or get_authoritative_registry()
    meta_p49 = reg.get("p49")
    base_p49 = meta_p49.baseline_value

    tfc = patient.total_follicle_count
    if tfc is None and patient.follicle_count_left is not None and patient.follicle_count_right is not None:
        tfc = patient.follicle_count_left + patient.follicle_count_right

    if tfc is None or math.isnan(tfc) or math.isinf(tfc) or tfc <= 0:
        final_val = base_p49
        audit = MappingAuditEntry(
            parameter_id="p49",
            parameter_name="k_AF1",
            clinical_marker="total_follicle_count",
            observed_value=None,
            reference_value=CLINICAL_REFERENCES["TOTAL_FOLLICLE_COUNT"],
            scale_factor=1.0,
            baseline_value=base_p49,
            final_value=final_val,
            status="missing_fallback",
            reason="Total follicle count observation is missing; assigned published baseline.",
        )
        return (final_val, audit)

    # Soft bounded scale from total follicle count
    scale_raw = bounded_scale(
        value=tfc,
        reference=CLINICAL_REFERENCES["TOTAL_FOLLICLE_COUNT"],
        max_delta=0.30,
        alpha=0.6,
        direction=1,
    )
    scale_net = max(0.65, min(scale_raw, 1.35))
    proposed_val = base_p49 * scale_net

    # Layer 2: Hard physiological safety clamp
    final_val = meta_p49.clamp(proposed_val)
    status = "clamped" if final_val != proposed_val else "personalized"
    reason = f"Personalized via total_follicle_count={tfc:.1f}; net scale={scale_net:.3f}."

    audit = MappingAuditEntry(
        parameter_id="p49",
        parameter_name="k_AF1",
        clinical_marker="total_follicle_count",
        observed_value=tfc,
        reference_value=CLINICAL_REFERENCES["TOTAL_FOLLICLE_COUNT"],
        scale_factor=scale_net,
        baseline_value=base_p49,
        final_value=final_val,
        status=status,
        reason=reason,
    )
    return (final_val, audit)


def map_p51_k_AF2_AF3(
    patient: "HPOPatientValues",
    registry: Optional[AuthoritativeParameterRegistry] = None,
) -> Tuple[float, MappingAuditEntry]:
    """
    Phase 7.9: Maps patient follicle maturation / cycle regularity to Röblitz parameter p51 (k_AF2_AF3).

    Biological Mechanism:
      Governs transition from antral follicles (AF2) to dominant pre-ovulatory stage (AF3) (re44).
      In PCOS, follicular arrest occurs at 2-9 mm with failure of dominant selection.
      Primary Signal: mean_follicle_size (reference median: 14.0 mm, max +/-25%, alpha=0.6).
      Secondary Signal: cycle_regular (irregular/amenorrheic cycle reduces transition rate by -10%).

    Layer 1 (Soft Bounds): Combined perturbation bounded to [-0.35, +0.35].
    Layer 2 (Hard Safety Clamp): Enforced via AuthoritativeParameterRegistry [0.0039, 0.0244].
    """
    reg = registry or get_authoritative_registry()
    meta_p51 = reg.get("p51")
    base_p51 = meta_p51.baseline_value

    size_val = patient.mean_follicle_size
    if size_val is None and patient.follicle_size_left is not None and patient.follicle_size_right is not None:
        size_val = (patient.follicle_size_left + patient.follicle_size_right) / 2.0

    cycle_reg = patient.cycle_regular

    # If both inputs are completely missing
    has_size = size_val is not None and not math.isnan(size_val) and not math.isinf(size_val) and size_val > 0
    has_cycle = cycle_reg is not None and not math.isnan(cycle_reg) and not math.isinf(cycle_reg)

    if not has_size and not has_cycle:
        final_val = base_p51
        audit = MappingAuditEntry(
            parameter_id="p51",
            parameter_name="k_AF2_AF3",
            clinical_marker="mean_follicle_size+cycle_regular",
            observed_value=None,
            reference_value=CLINICAL_REFERENCES["MEAN_FOLLICLE_SIZE"],
            scale_factor=1.0,
            baseline_value=base_p51,
            final_value=final_val,
            status="missing_fallback",
            reason="Follicle size and cycle regularity unobserved; assigned published baseline.",
        )
        return (final_val, audit)

    # 1. Primary scale from mean follicle size (direction=+1, smaller size -> lower rate)
    if has_size:
        s_size = bounded_scale(
            value=size_val,
            reference=CLINICAL_REFERENCES["MEAN_FOLLICLE_SIZE"],
            max_delta=0.25,
            alpha=0.6,
            direction=1,
        )
        delta_size = s_size - 1.0
    else:
        delta_size = 0.0

    # 2. Secondary modulation from cycle regularity
    # In clinical dataset: 2.0 = Regular, >=3.0 (4.0/5.0) = Irregular
    if has_cycle and cycle_reg >= 3.0:
        delta_cycle = -0.10  # Follicular arrest / irregular cycle attenuates dominant follicle transition
    else:
        delta_cycle = 0.0

    total_delta = max(-0.35, min(delta_size + delta_cycle, 0.35))
    scale_net = 1.0 + total_delta
    proposed_val = base_p51 * scale_net

    # Layer 2: Hard physiological safety clamp
    final_val = meta_p51.clamp(proposed_val)
    status = "clamped" if final_val != proposed_val else "personalized"

    size_str = f"{size_val:.1f}mm" if has_size else "N/A"
    cycle_str = "Irregular" if (has_cycle and cycle_reg >= 3.0) else ("Regular" if has_cycle else "N/A")
    reason = (
        f"Personalized via mean_size={size_str} (delta={delta_size:+.3f}) and "
        f"cycle={cycle_str} (delta={delta_cycle:+.3f}); net scale={scale_net:.3f}."
    )

    audit = MappingAuditEntry(
        parameter_id="p51",
        parameter_name="k_AF2_AF3",
        clinical_marker="mean_follicle_size+cycle_regular",
        observed_value=size_val if has_size else None,
        reference_value=CLINICAL_REFERENCES["MEAN_FOLLICLE_SIZE"],
        scale_factor=scale_net,
        baseline_value=base_p51,
        final_value=final_val,
        status=status,
        reason=reason,
    )
    return (final_val, audit)


def map_p173_b_IhB(
    patient: "HPOPatientValues",
    registry: Optional[AuthoritativeParameterRegistry] = None,
) -> Tuple[float, MappingAuditEntry]:
    """
    Phase 7.9: Maps patient follicle count and AMH to Röblitz parameter p173 (b_IhB).

    Biological Mechanism:
      Governs basal synthesis of Inhibin B (re72 -> InhB).
      Inhibin B and AMH are secreted by granulosa cells of early antral follicles.
      Higher follicle count and elevated AMH directly indicate expanded granulosa cell mass
      and increased Inhibin B hypersecretion.
      Primary Signal: total_follicle_count (reference: 12.0 follicles, max +/-25%, alpha=0.6).
      Secondary Signal: serum AMH (reference: 2.5 ng/mL, max +/-10%, alpha=0.5).

    Layer 1 (Soft Bounds): Combined perturbation bounded to [-0.35, +0.35].
    Layer 2 (Hard Safety Clamp): Enforced via AuthoritativeParameterRegistry [35.7974, 223.7337].
    """
    reg = registry or get_authoritative_registry()
    meta_p173 = reg.get("p173")
    base_p173 = meta_p173.baseline_value

    tfc = patient.total_follicle_count
    if tfc is None and patient.follicle_count_left is not None and patient.follicle_count_right is not None:
        tfc = patient.follicle_count_left + patient.follicle_count_right

    amh_val = patient.amh

    has_tfc = tfc is not None and not math.isnan(tfc) and not math.isinf(tfc) and tfc > 0
    has_amh = amh_val is not None and not math.isnan(amh_val) and not math.isinf(amh_val) and amh_val > 0

    if not has_tfc and not has_amh:
        final_val = base_p173
        audit = MappingAuditEntry(
            parameter_id="p173",
            parameter_name="b_IhB",
            clinical_marker="total_follicle_count+amh",
            observed_value=None,
            reference_value=CLINICAL_REFERENCES["TOTAL_FOLLICLE_COUNT"],
            scale_factor=1.0,
            baseline_value=base_p173,
            final_value=final_val,
            status="missing_fallback",
            reason="Follicle count and AMH unobserved; assigned published baseline.",
        )
        return (final_val, audit)

    # 1. Primary scale from follicle count (max_delta=0.25, alpha=0.6)
    if has_tfc:
        s_tfc = bounded_scale(
            value=tfc,
            reference=CLINICAL_REFERENCES["TOTAL_FOLLICLE_COUNT"],
            max_delta=0.25,
            alpha=0.6,
            direction=1,
        )
        delta_tfc = s_tfc - 1.0
    else:
        delta_tfc = 0.0

    # 2. Secondary scale from AMH (max_delta=0.10, alpha=0.5)
    if has_amh:
        s_amh = bounded_scale(
            value=amh_val,
            reference=CLINICAL_REFERENCES["AMH"],
            max_delta=0.10,
            alpha=0.5,
            direction=1,
        )
        delta_amh = s_amh - 1.0
    else:
        delta_amh = 0.0

    total_delta = max(-0.35, min(delta_tfc + delta_amh, 0.35))
    scale_net = 1.0 + total_delta
    proposed_val = base_p173 * scale_net

    # Layer 2: Hard physiological safety clamp
    final_val = meta_p173.clamp(proposed_val)
    status = "clamped" if final_val != proposed_val else "personalized"

    tfc_str = f"{tfc:.1f}" if has_tfc else "N/A"
    amh_str = f"{amh_val:.2f}ng/mL" if has_amh else "N/A"
    reason = (
        f"Personalized via follicle_count={tfc_str} (delta={delta_tfc:+.3f}) and "
        f"AMH={amh_str} (delta={delta_amh:+.3f}); net scale={scale_net:.3f}."
    )

    audit = MappingAuditEntry(
        parameter_id="p173",
        parameter_name="b_IhB",
        clinical_marker="total_follicle_count+amh",
        observed_value=tfc if has_tfc else amh_val,
        reference_value=CLINICAL_REFERENCES["TOTAL_FOLLICLE_COUNT"],
        scale_factor=scale_net,
        baseline_value=base_p173,
        final_value=final_val,
        status=status,
        reason=reason,
    )
    return (final_val, audit)


def map_core_5_parameters(
    patient: "HPOPatientValues",
    registry: Optional[AuthoritativeParameterRegistry] = None,
) -> Tuple[Dict[str, float], List[MappingAuditEntry]]:
    """
    Phase 7.9: Maps all 5 validated Core Personalizable parameters simultaneously:
      - p1   (b_syn_LH)   <- Serum LH & FSH/LH ratio
      - p17  (b_rel_FSH)  <- Serum FSH & FSH/LH ratio
      - p49  (k_AF1)      <- Total follicle count
      - p51  (k_AF2_AF3)  <- Mean follicle size & Cycle regularity
      - p173 (b_IhB)      <- Total follicle count & AMH
    """
    reg = registry or get_authoritative_registry()
    ode_params: Dict[str, float] = {}
    audit_log: List[MappingAuditEntry] = []

    # 1. p1 (b_syn_LH)
    p1_val, p1_audit = map_p1_b_syn_LH(patient, reg)
    ode_params["p1"] = p1_val
    ode_params["b_syn_LH"] = p1_val
    audit_log.append(p1_audit)

    # 2. p17 (b_rel_FSH)
    p17_val, p17_audit = map_p17_b_rel_FSH(patient, reg)
    ode_params["p17"] = p17_val
    ode_params["b_rel_FSH"] = p17_val
    audit_log.append(p17_audit)

    # 3. p49 (k_AF1)
    p49_val, p49_audit = map_p49_k_AF1(patient, reg)
    ode_params["p49"] = p49_val
    ode_params["k_AF1"] = p49_val
    audit_log.append(p49_audit)

    # 4. p51 (k_AF2_AF3)
    p51_val, p51_audit = map_p51_k_AF2_AF3(patient, reg)
    ode_params["p51"] = p51_val
    ode_params["k_AF2_AF3"] = p51_val
    audit_log.append(p51_audit)

    # 5. p173 (b_IhB)
    p173_val, p173_audit = map_p173_b_IhB(patient, reg)
    ode_params["p173"] = p173_val
    ode_params["b_IhB"] = p173_val
    audit_log.append(p173_audit)

    return (ode_params, audit_log)



