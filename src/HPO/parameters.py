"""
parameters.py

Authoritative classification and registry for all 141 Röblitz 2013 HPO model parameters.
Implements Phase 7.3:
  - Categorization into STRUCTURAL, PERSONALIZABLE_CORE, EXPLORATORY, and DRUG_PK.
  - Hard physiological safety bounds [min_bound, max_bound].
  - Bidirectional resolution (ID <-> Name) and clamping utilities.
  - Backwards-compatible HPOParameters dataclass.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math

from src.HPO.sbml_parameters import get_sbml_parameter_registry

ROOT_DIR = Path(__file__).resolve().parents[2]


class ParameterCategory(str, Enum):
    """Classification of Röblitz model parameters."""
    PERSONALIZABLE_CORE = "personalizable_core"  # Tier 1: Core 5 parameters for Phase 7
    EXPLORATORY = "exploratory"                  # Tier 2: Candidate parameters for Phase 8 calibration
    DRUG_PK = "drug_pk"                          # GnRH agonist/antagonist pharmacology constants
    STRUCTURAL = "structural"                    # Invariant physical/biochemical constants (Hill coeff, blood vol, etc.)


@dataclass(frozen=True)
class ParameterMetadata:
    """Detailed metadata, classification, and safety bounds for an HPO model parameter."""
    id: str
    name: str
    baseline_value: float
    category: ParameterCategory
    min_bound: float
    max_bound: float
    description: str = ""

    @property
    def is_personalizable(self) -> bool:
        return self.category in (ParameterCategory.PERSONALIZABLE_CORE, ParameterCategory.EXPLORATORY)

    def clamp(self, value: float) -> float:
        """Clamp a value to stay strictly within hard physiological bounds."""
        if math.isnan(value) or math.isinf(value):
            return self.baseline_value
        return max(self.min_bound, min(value, self.max_bound))


class AuthoritativeParameterRegistry:
    """
    Authoritative registry mapping all 141 Röblitz SBML parameters to their
    verified classifications, descriptions, and hard safety bounds.
    """

    # Core 5 parameters for Phase 7
    CORE_SPECS = {
        "p1": {
            "name": "b_syn_LH",
            "desc": "Basal LH synthesis rate in pituitary gland.",
            "min_ratio": 0.40,
            "max_ratio": 2.50,
        },
        "p17": {
            "name": "b_rel_FSH",
            "desc": "Basal FSH release rate from pituitary into circulation.",
            "min_ratio": 0.40,
            "max_ratio": 2.50,
        },
        "p49": {
            "name": "k_AF1",
            "desc": "Maximal recruitment rate into early antral follicles (AF1).",
            "min_ratio": 0.40,
            "max_ratio": 2.50,
        },
        "p51": {
            "name": "k_AF2_AF3",
            "desc": "Follicle transition/selection rate from AF2 to dominant AF3.",
            "min_ratio": 0.40,
            "max_ratio": 2.50,
        },
        "p173": {
            "name": "b_IhB",
            "desc": "Basal synthesis rate of circulating Inhibin B.",
            "min_ratio": 0.40,
            "max_ratio": 2.50,
        },
    }

    # Exploratory candidates for Phase 8 calibration
    EXPLORATORY_SPECS = {
        "p16": ("b_rel_LH", "Basal LH release rate from pituitary into bloodstream."),
        "p50": ("k_AF1_AF2", "Follicular progression rate from AF1 to AF2 stage."),
        "p158": ("b_E2", "Basal Estradiol (E2) synthesis rate."),
        "p166": ("b_P4", "Basal Progesterone (P4) synthesis rate."),
        "p168": ("b_IhA", "Basal Inhibin A (IhA) synthesis rate."),
        "p174": ("k_AF2_IhB", "Inhibin B production rate from AF2 antral follicles."),
    }

    def __init__(self):
        self._by_id: Dict[str, ParameterMetadata] = {}
        self._by_name: Dict[str, ParameterMetadata] = {}
        self._parameter_list: List[ParameterMetadata] = []

        self._build_registry()

    def _build_registry(self) -> None:
        sbml_reg = get_sbml_parameter_registry()

        for param in sbml_reg.list_parameters():
            pid = param.id
            pname = param.name
            base_val = param.value

            # Determine category, bounds, and description
            if pid in self.CORE_SPECS or pname in self.CORE_SPECS:
                spec = self.CORE_SPECS.get(pid) or self.CORE_SPECS[pname]
                cat = ParameterCategory.PERSONALIZABLE_CORE
                min_b = base_val * spec["min_ratio"]
                max_b = base_val * spec["max_ratio"]
                desc = spec["desc"]

            elif pid in self.EXPLORATORY_SPECS or pname in self.EXPLORATORY_SPECS:
                spec_name, desc = self.EXPLORATORY_SPECS.get(pid) or self.EXPLORATORY_SPECS[pname]
                cat = ParameterCategory.EXPLORATORY
                min_b = base_val * 0.40
                max_b = base_val * 2.50

            elif any(token in pname for token in ("Ago", "Ant", "ago", "ant")):
                cat = ParameterCategory.DRUG_PK
                min_b = base_val
                max_b = base_val
                desc = "Pharmacokinetic parameter for GnRH agonist/antagonist administration."

            else:
                cat = ParameterCategory.STRUCTURAL
                min_b = base_val
                max_b = base_val
                desc = "Structural biochemical/physiological invariant constant."

            metadata = ParameterMetadata(
                id=pid,
                name=pname,
                baseline_value=base_val,
                category=cat,
                min_bound=min_b,
                max_bound=max_b,
                description=desc,
            )

            self._by_id[pid] = metadata
            self._by_name[pname] = metadata
            self._parameter_list.append(metadata)

    def __len__(self) -> int:
        return len(self._parameter_list)

    def __contains__(self, key: str) -> bool:
        return key in self._by_id or key in self._by_name

    def __getitem__(self, key: str) -> ParameterMetadata:
        return self.get(key)

    def get(self, key: str) -> ParameterMetadata:
        """Lookup parameter metadata by SBML ID or biological name."""
        if key in self._by_id:
            return self._by_id[key]
        if key in self._by_name:
            return self._by_name[key]
        raise KeyError(
            f"Unknown Röblitz parameter '{key}'. "
            f"Expected a valid SBML ID (e.g. 'p1') or biological name (e.g. 'b_syn_LH')."
        )

    def get_core_personalizable(self) -> Dict[str, ParameterMetadata]:
        """Return the 5 Tier-1 Core personalizable parameters for Phase 7."""
        return {
            pid: self._by_id[pid]
            for pid in self.CORE_SPECS
            if pid in self._by_id
        }

    def get_by_category(self, category: ParameterCategory) -> List[ParameterMetadata]:
        """Return all parameters belonging to a specific category."""
        return [p for p in self._parameter_list if p.category == category]

    def get_bounds(self, key: str) -> Tuple[float, float]:
        """Return the hard safety bounds (min_bound, max_bound) for a parameter."""
        meta = self.get(key)
        return (meta.min_bound, meta.max_bound)

    def clamp(self, key: str, value: float) -> float:
        """Clamp a proposed parameter value within its hard safety bounds."""
        return self.get(key).clamp(value)

    def list_all(self) -> List[ParameterMetadata]:
        """Return all 141 parameters as a list."""
        return list(self._parameter_list)

    def export_csv(self, output_path: Path | str) -> Path:
        """Export classification and safety bounds to a CSV file for documentation."""
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("id,name,baseline_value,category,min_bound,max_bound,description\n")
            for p in self._parameter_list:
                desc_clean = p.description.replace(",", ";")
                f.write(
                    f"{p.id},{p.name},{p.baseline_value},{p.category.value},"
                    f"{p.min_bound},{p.max_bound},\"{desc_clean}\"\n"
                )

        return out_path


_AUTH_REGISTRY: Optional[AuthoritativeParameterRegistry] = None


def get_authoritative_registry() -> AuthoritativeParameterRegistry:
    """Return singleton instance of AuthoritativeParameterRegistry."""
    global _AUTH_REGISTRY
    if _AUTH_REGISTRY is None:
        _AUTH_REGISTRY = AuthoritativeParameterRegistry()
    return _AUTH_REGISTRY


# =====================================================================
# Backwards-compatible Legacy Dataclass
# =====================================================================

@dataclass
class HPOParameters:
    """
    Physiological model parameters (legacy dataclass preserved for backwards compatibility).
    """

    # Core axis
    gnrh_production: Optional[float] = None
    gnrh_clearance: Optional[float] = None

    fsh_production: Optional[float] = None
    fsh_clearance: Optional[float] = None

    lh_production: Optional[float] = None
    lh_clearance: Optional[float] = None

    # Ovarian hormones
    estradiol_production: Optional[float] = None
    estradiol_clearance: Optional[float] = None

    progesterone_production: Optional[float] = None
    progesterone_clearance: Optional[float] = None

    # Inhibins
    inhibin_a_production: Optional[float] = None
    inhibin_a_clearance: Optional[float] = None

    inhibin_b_production: Optional[float] = None
    inhibin_b_clearance: Optional[float] = None

    # Feedback
    fsh_feedback: Optional[float] = None
    lh_feedback: Optional[float] = None

    estradiol_feedback: Optional[float] = None
    progesterone_feedback: Optional[float] = None

    # Follicular dynamics
    follicular_growth_rate: Optional[float] = None
    follicular_atresia_rate: Optional[float] = None

    # Patient-specific modifiers
    amh_modifier: Optional[float] = None
    metabolic_modifier: Optional[float] = None
    cycle_modifier: Optional[float] = None