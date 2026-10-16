"""
sbml_parameters.py

Source of truth for the 141 Röblitz et al. 2013 HPO model parameters.
Directly extracts, verifies, and indexes parameters from the published SBML file
(models/BIOMD0000000494.xml).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import math
import libsbml


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SBML_PATH = ROOT_DIR / "models" / "BIOMD0000000494.xml"


@dataclass(frozen=True)
class SBMLParameter:
    """Represents a single parameter defined in the Röblitz 2013 SBML model."""
    id: str
    name: str
    value: float
    units: str
    constant: bool

    def __repr__(self) -> str:
        return f"SBMLParameter(id='{self.id}', name='{self.name}', value={self.value})"


class SBMLParameterRegistry:
    """
    Authoritative registry of Röblitz 2013 SBML parameters.

    Provides bidirectional lookup:
      - By SBML ID (e.g. 'p1', 'p17', 'p49')
      - By biological name (e.g. 'b_syn_LH', 'b_rel_FSH', 'k_AF1')
    """

    EXPECTED_PARAMETER_COUNT = 141

    # Tier 1 Core Candidates identified for Phase 7 personalization
    CORE_CANDIDATES = {
        "p1": ("b_syn_LH", 7309.91587614104),
        "p17": ("b_rel_FSH", 0.0569894397708967),
        "p49": ("k_AF1", 3.66180418829425),
        "p51": ("k_AF2_AF3", 4.88231609092536),
        "p173": ("b_IhB", 89.4934669769107),
    }

    def __init__(self, sbml_path: Optional[Path | str] = None):
        self.sbml_path = Path(sbml_path) if sbml_path else DEFAULT_SBML_PATH
        self._by_id: Dict[str, SBMLParameter] = {}
        self._by_name: Dict[str, SBMLParameter] = {}
        self._parameters_list: List[SBMLParameter] = []

        self._load_and_validate()

    def _load_and_validate(self) -> None:
        if not self.sbml_path.exists():
            raise FileNotFoundError(
                f"SBML model file not found at '{self.sbml_path}'.\n"
                f"Please ensure models/BIOMD0000000494.xml exists."
            )

        reader = libsbml.SBMLReader()
        document = reader.readSBMLFromFile(str(self.sbml_path))

        if document.getNumErrors() > 0:
            errors = [
                document.getError(i).getMessage()
                for i in range(document.getNumErrors())
                if document.getError(i).isError() or document.getError(i).isFatal()
            ]
            if errors:
                raise RuntimeError(
                    f"Errors while parsing SBML file '{self.sbml_path}':\n"
                    + "\n".join(errors)
                )

        model = document.getModel()
        if model is None:
            raise RuntimeError(f"Could not load SBML model from '{self.sbml_path}'.")

        num_params = model.getNumParameters()
        if num_params != self.EXPECTED_PARAMETER_COUNT:
            raise ValueError(
                f"Expected {self.EXPECTED_PARAMETER_COUNT} parameters in SBML model, "
                f"but found {num_params}."
            )

        for i in range(num_params):
            param = model.getParameter(i)
            param_id = param.getId()
            param_name = param.getName() or param_id
            param_value = param.getValue()
            param_units = param.getUnits() or ""
            is_constant = param.getConstant()

            if math.isnan(param_value) or math.isinf(param_value):
                raise ValueError(
                    f"Parameter '{param_id}' ('{param_name}') has non-finite value: {param_value}"
                )

            sbml_param = SBMLParameter(
                id=param_id,
                name=param_name,
                value=float(param_value),
                units=param_units,
                constant=is_constant,
            )

            self._by_id[param_id] = sbml_param
            self._by_name[param_name] = sbml_param
            self._parameters_list.append(sbml_param)

    def __len__(self) -> int:
        return len(self._parameters_list)

    def __contains__(self, key: str) -> bool:
        return key in self._by_id or key in self._by_name

    def __getitem__(self, key: str) -> SBMLParameter:
        return self.get(key)

    def get(self, key: str) -> SBMLParameter:
        """Retrieve a parameter by its SBML ID or biological name."""
        if key in self._by_id:
            return self._by_id[key]
        if key in self._by_name:
            return self._by_name[key]
        raise KeyError(
            f"Unknown Röblitz parameter '{key}'. "
            f"Expected a valid SBML ID (e.g. 'p1') or biological name (e.g. 'b_syn_LH')."
        )

    def resolve_id(self, key: str) -> str:
        """Resolve a key (ID or name) to its SBML ID."""
        return self.get(key).id

    def resolve_name(self, key: str) -> str:
        """Resolve a key (ID or name) to its biological name."""
        return self.get(key).name

    def get_baseline_value(self, key: str) -> float:
        """Return the baseline value for an ID or name."""
        return self.get(key).value

    def list_parameters(self) -> List[SBMLParameter]:
        """Return all 141 parameters in order of definition in the SBML model."""
        return list(self._parameters_list)

    def to_dict(self, by: str = "id") -> Dict[str, float]:
        """Return a dictionary of {key: baseline_value} where key is 'id' or 'name'."""
        if by == "id":
            return {p.id: p.value for p in self._parameters_list}
        elif by == "name":
            return {p.name: p.value for p in self._parameters_list}
        else:
            raise ValueError(f"Invalid 'by' argument: {by}. Must be 'id' or 'name'.")

    def verify_core_candidates(self) -> Dict[str, bool]:
        """Verify presence and baseline values of the Core 5 candidate parameters."""
        results = {}
        for param_id, (expected_name, expected_val) in self.CORE_CANDIDATES.items():
            param = self.get(param_id)
            name_matches = param.name == expected_name
            val_matches = math.isclose(param.value, expected_val, rel_tol=1e-5)
            results[param_id] = name_matches and val_matches
        return results

    def export_csv(self, output_path: Path | str) -> Path:
        """Export all 141 parameters to a CSV file for auditing."""
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("id,name,value,units,constant\n")
            for p in self._parameters_list:
                f.write(f"{p.id},{p.name},{p.value},{p.units},{p.constant}\n")

        return out_path


_REGISTRY_INSTANCE: Optional[SBMLParameterRegistry] = None


def get_sbml_parameter_registry() -> SBMLParameterRegistry:
    """Return a singleton instance of SBMLParameterRegistry."""
    global _REGISTRY_INSTANCE
    if _REGISTRY_INSTANCE is None:
        _REGISTRY_INSTANCE = SBMLParameterRegistry()
    return _REGISTRY_INSTANCE
