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

