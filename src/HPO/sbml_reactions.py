"""
sbml_reactions.py

Inspects, parses, and validates the kinetic rate laws and reactions
in the Röblitz 2013 SBML model (models/BIOMD0000000494.xml).
Verifies the exact biological and mathematical role of candidate parameters.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set
import re
import libsbml

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SBML_PATH = ROOT_DIR / "models" / "BIOMD0000000494.xml"


@dataclass(frozen=True)
class ReactionParticipant:
    """Represents a reaction where a target parameter participates in the kinetic law."""
    reaction_id: str
    reaction_name: str
    parameter_id: str
    parameter_name: str
    reactants: List[str] = field(default_factory=list)
    products: List[str] = field(default_factory=list)
    modifiers: List[str] = field(default_factory=list)
    rate_law_call: str = ""
    function_id: str = ""
    function_body: str = ""
    biological_role: str = ""


class SBMLReactionAnalyzer:
    """
    Parses and audits reactions and kinetic laws in the Röblitz 2013 SBML model.
    """

    EXPECTED_REACTION_COUNT = 71

    # Verified role annotations for the Tier-1 Core Candidates
    CORE_REACTION_SPECS = {
        "p1": {
            "name": "b_syn_LH",
            "primary_reaction": "re25",
            "target_product": "LH_Pit",
            "role": "Basal LH synthesis into the pituitary LH reservoir.",
        },
        "p17": {
            "name": "b_rel_FSH",
            "primary_reaction": "re84",
            "target_product": "FSH_bld",
            "role": "Basal release of stored pituitary FSH into systemic blood circulation.",
        },
        "p49": {
            "name": "k_AF1",
            "primary_reaction": "re43",
            "target_product": "AF1",
            "role": "Maximal recruitment rate into the early antral follicle pool (AF1).",
        },
        "p51": {
            "name": "k_AF2_AF3",
            "primary_reaction": "re44",
            "target_product": "AF3",
            "role": "Kinetic rate constant governing transition/selection from AF2 to AF3 follicle stage.",
        },
        "p173": {
            "name": "b_IhB",
            "primary_reaction": "re72",
            "target_product": "InhB",
            "role": "Basal synthesis rate of circulating Inhibin B.",
        },
    }

    def __init__(self, sbml_path: Optional[Path | str] = None):
        self.sbml_path = Path(sbml_path) if sbml_path else DEFAULT_SBML_PATH
        self._reactions_by_param: Dict[str, List[ReactionParticipant]] = {}
        self._all_reactions: Dict[str, dict] = {}
        self._function_definitions: Dict[str, str] = {}

        self._parse_and_audit()

    def _parse_and_audit(self) -> None:
        if not self.sbml_path.exists():
            raise FileNotFoundError(
                f"SBML model file not found at '{self.sbml_path}'."
            )

        reader = libsbml.SBMLReader()
        document = reader.readSBMLFromFile(str(self.sbml_path))
        model = document.getModel()
        if model is None:
            raise RuntimeError(f"Could not load SBML model from '{self.sbml_path}'.")

        # 1. Map function definitions
        for i in range(model.getNumFunctionDefinitions()):
            fd = model.getFunctionDefinition(i)
            self._function_definitions[fd.getId()] = libsbml.formulaToString(fd.getMath())

        # 2. Iterate and inspect all reactions
        num_reactions = model.getNumReactions()
        if num_reactions != self.EXPECTED_REACTION_COUNT:
            raise ValueError(
                f"Expected {self.EXPECTED_REACTION_COUNT} reactions in SBML model, "
                f"but found {num_reactions}."
            )

        for i in range(num_reactions):
            r = model.getReaction(i)
            rid = r.getId()
            rname = r.getName() or rid

            reactants = [r.getReactant(j).getSpecies() for j in range(r.getNumReactants())]
            products = [r.getProduct(j).getSpecies() for j in range(r.getNumProducts())]
            # Remove duplicates from modifiers while preserving order
            seen_mods: Set[str] = set()
            modifiers = []
            for j in range(r.getNumModifiers()):
                sp = r.getModifier(j).getSpecies()
                if sp not in seen_mods:
                    seen_mods.add(sp)
                    modifiers.append(sp)

            kl = r.getKineticLaw()
            rate_law = kl.getFormula() if kl else ""

            # Extract function call if present
            match_fn = re.search(r"(function_\d+)\((.*?)\)", rate_law)
            fn_id = match_fn.group(1) if match_fn else ""
            fn_body = self._function_definitions.get(fn_id, "")

            self._all_reactions[rid] = {
                "id": rid,
                "name": rname,
                "reactants": reactants,
                "products": products,
                "modifiers": modifiers,
                "rate_law": rate_law,
                "function_id": fn_id,
                "function_body": fn_body,
            }

            # Find all whole-word tokens in kinetic law
            tokens = set(re.findall(r"\b[a-zA-Z0-9_]+\b", rate_law))

            for pid, spec in self.CORE_REACTION_SPECS.items():
                if pid in tokens:
                    participant = ReactionParticipant(
                        reaction_id=rid,
                        reaction_name=rname,
                        parameter_id=pid,
                        parameter_name=spec["name"],
                        reactants=reactants,
                        products=products,
                        modifiers=modifiers,
                        rate_law_call=rate_law,
                        function_id=fn_id,
                        function_body=fn_body,
                        biological_role=spec["role"],
                    )
                    self._reactions_by_param.setdefault(pid, []).append(participant)

    def get_reactions_for_parameter(self, param_id: str) -> List[ReactionParticipant]:
        """Return all reactions where a specific parameter participates in the rate law."""
        return self._reactions_by_param.get(param_id, [])

    def verify_core_candidate_reactions(self) -> Dict[str, bool]:
        """
        Verify that each of the 5 Core Candidate parameters participates
        in its expected primary reaction and affects its expected target species.
        """
        results = {}
        for pid, spec in self.CORE_REACTION_SPECS.items():
            participants = self.get_reactions_for_parameter(pid)
            rxn_ids = [p.reaction_id for p in participants]
            has_primary_rxn = spec["primary_reaction"] in rxn_ids

            # Check target product species
            all_products = []
            for p in participants:
                all_products.extend(p.products)
            has_target_product = spec["target_product"] in all_products

            results[pid] = has_primary_rxn and has_target_product

        return results

    def export_csv(self, output_path: Path | str) -> Path:
        """Export candidate reaction audit to a CSV file for documentation."""
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("parameter_id,parameter_name,reaction_id,reactants,products,modifiers,function_id,biological_role\n")
            for pid, participants in self._reactions_by_param.items():
                for p in participants:
                    r_str = ";".join(p.reactants)
                    p_str = ";".join(p.products)
                    m_str = ";".join(p.modifiers)
                    role = p.biological_role.replace(",", ";")
                    f.write(
                        f"{p.parameter_id},{p.parameter_name},{p.reaction_id},"
                        f"\"{r_str}\",\"{p_str}\",\"{m_str}\",{p.function_id},\"{role}\"\n"
                    )

        return out_path


_ANALYZER_INSTANCE: Optional[SBMLReactionAnalyzer] = None


def get_sbml_reaction_analyzer() -> SBMLReactionAnalyzer:
    """Return a singleton instance of SBMLReactionAnalyzer."""
    global _ANALYZER_INSTANCE
    if _ANALYZER_INSTANCE is None:
        _ANALYZER_INSTANCE = SBMLReactionAnalyzer()
    return _ANALYZER_INSTANCE
