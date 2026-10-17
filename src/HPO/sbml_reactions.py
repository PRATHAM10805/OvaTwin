"""Read reaction metadata from the active Röblitz SBML model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import zipfile

import libsbml

from src.HPO.model_runner import OMEX_PATH, SBML_FILENAME


EXPECTED_REACTION_COUNT = 71


@dataclass(frozen=True)
class ReactionParticipant:
    """A species participating in a reaction."""

    species_id: str
    species_name: str
    stoichiometry: float


@dataclass(frozen=True)
class SBMLReaction:
    """Metadata for one SBML reaction."""

    id: str
    name: str
    reversible: bool
    reactants: tuple[ReactionParticipant, ...]
    products: tuple[ReactionParticipant, ...]
    modifiers: tuple[tuple[str, str], ...]
    kinetic_law: str
    direct_parameter_refs: tuple[str, ...]
    direct_species_refs: tuple[str, ...]


class SBMLReactionRegistry:
    """Read-only registry of reactions in the active Röblitz model."""

    def __init__(
        self,
        omex_path: Path | str = OMEX_PATH,
        sbml_filename: str = SBML_FILENAME,
        expected_count: Optional[int] = EXPECTED_REACTION_COUNT,
    ) -> None:
        self.omex_path = Path(omex_path)
        self.sbml_filename = sbml_filename

        self._by_id: dict[str, SBMLReaction] = {}
        self._reactions: list[SBMLReaction] = []
        self._species_ids: set[str] = set()

        self._load(expected_count)

    def _read_sbml_model(self):
        """Load the exact SBML archive entry used by the simulation runner."""

        if not self.omex_path.is_file():
            raise FileNotFoundError(
                f"Röblitz OMEX file not found: {self.omex_path}"
            )

        try:
            with zipfile.ZipFile(self.omex_path) as archive:
                if self.sbml_filename not in archive.namelist():
                    raise FileNotFoundError(
                        f"SBML entry {self.sbml_filename!r} not found "
                        f"inside {self.omex_path}"
                    )

                xml_text = archive.read(
                    self.sbml_filename
                ).decode("utf-8")

        except zipfile.BadZipFile as exc:
            raise RuntimeError(
                f"Invalid OMEX archive: {self.omex_path}"
            ) from exc

        document = libsbml.SBMLReader().readSBMLFromString(xml_text)

        errors = [
            document.getError(i).getMessage()
            for i in range(document.getNumErrors())
            if (
                document.getError(i).isError()
                or document.getError(i).isFatal()
            )
        ]

        if errors:
            raise RuntimeError(
                "Could not parse active Röblitz SBML:\n"
                + "\n".join(errors)
            )

        model = document.getModel()

        if model is None:
            raise RuntimeError("The SBML document contains no model.")

        return model

    @staticmethod
    def _collect_ast_names(node) -> set[str]:
        """Collect variable names from an SBML MathML syntax tree."""

        names: set[str] = set()

        if node is None:
            return names

        # Use the ASTNode method rather than repeatedly accessing
        # libsbml.AST_NAME.
        if node.isName():
            name = node.getName()

            if name:
                names.add(name)

        for index in range(node.getNumChildren()):
            child = node.getChild(index)
            names.update(
                SBMLReactionRegistry._collect_ast_names(child)
            )

        return names

    @staticmethod
    def _participant_from_reference(
        model,
        reference,
    ) -> ReactionParticipant:
        species_id = reference.getSpecies()
        species = model.getSpecies(species_id)

        species_name = (
            species.getName() or species_id
            if species is not None
            else species_id
        )

        return ReactionParticipant(
            species_id=species_id,
            species_name=species_name,
            stoichiometry=float(reference.getStoichiometry()),
        )

    def _load(self, expected_count: Optional[int]) -> None:
        model = self._read_sbml_model()

        count = model.getNumReactions()

        if expected_count is not None and count != expected_count:
            raise ValueError(
                f"Expected {expected_count} reactions, found {count}."
            )

        parameter_ids = {
            model.getParameter(i).getId()
            for i in range(model.getNumParameters())
        }

        self._species_ids = {
            model.getSpecies(i).getId()
            for i in range(model.getNumSpecies())
        }

        for index in range(count):
            reaction = model.getReaction(index)
            reaction_id = reaction.getId()

            if not reaction_id:
                raise ValueError(
                    f"Reaction at index {index} has no ID."
                )

            if reaction_id in self._by_id:
                raise ValueError(
                    f"Duplicate reaction ID: {reaction_id!r}"
                )

            reactants = tuple(
                self._participant_from_reference(model, reference)
                for reference in (
                    reaction.getReactant(i)
                    for i in range(reaction.getNumReactants())
                )
            )

            products = tuple(
                self._participant_from_reference(model, reference)
                for reference in (
                    reaction.getProduct(i)
                    for i in range(reaction.getNumProducts())
                )
            )

            modifiers = tuple(
                (
                    reaction.getModifier(i).getSpecies(),
                    (
                        model.getSpecies(
                            reaction.getModifier(i).getSpecies()
                        ).getName()
                        or reaction.getModifier(i).getSpecies()
                    ),
                )
                for i in range(reaction.getNumModifiers())
            )

            kinetic_law = ""
            direct_parameter_refs: set[str] = set()
            direct_species_refs: set[str] = set()

            law = reaction.getKineticLaw()

            if law is not None and law.getMath() is not None:
                math_node = law.getMath()

                kinetic_law = (
                    libsbml.formulaToL3String(math_node)
                    or libsbml.formulaToString(math_node)
                    or ""
                )

                names = self._collect_ast_names(math_node)

                direct_parameter_refs = names & parameter_ids
                direct_species_refs = names & self._species_ids

            item = SBMLReaction(
                id=reaction_id,
                name=reaction.getName() or reaction_id,
                reversible=bool(reaction.getReversible()),
                reactants=reactants,
                products=products,
                modifiers=modifiers,
                kinetic_law=kinetic_law,
                direct_parameter_refs=tuple(
                    sorted(direct_parameter_refs)
                ),
                direct_species_refs=tuple(
                    sorted(direct_species_refs)
                ),
            )

            self._by_id[item.id] = item
            self._reactions.append(item)

    def __len__(self) -> int:
        return len(self._reactions)

    def __contains__(self, reaction_id: str) -> bool:
        return reaction_id in self._by_id

    def __getitem__(self, reaction_id: str) -> SBMLReaction:
        return self.get(reaction_id)

    def get(self, reaction_id: str) -> SBMLReaction:
        """Retrieve a reaction by its SBML ID."""

        try:
            return self._by_id[reaction_id]
        except KeyError as exc:
            raise KeyError(
                f"Unknown Röblitz reaction: {reaction_id!r}"
            ) from exc

    def list_reactions(self) -> list[SBMLReaction]:
        """Return reactions in their SBML definition order."""

        return list(self._reactions)

    def list_species_ids(self) -> set[str]:
        """Return IDs of species defined in the model."""

        return set(self._species_ids)

    def find_by_parameter(self, parameter_id: str) -> list[SBMLReaction]:
        """
        Find reactions whose rate law directly references a parameter ID.

        This does not expand custom SBML function definitions. Therefore,
        an empty result does not prove that a parameter has no indirect
        influence on a reaction.
        """

        return [
            reaction
            for reaction in self._reactions
            if parameter_id in reaction.direct_parameter_refs
        ]

    def find_by_species(self, species_id: str) -> list[SBMLReaction]:
        """Find reactions that use a species as a reactant or product."""

        return [
            reaction
            for reaction in self._reactions
            if (
                any(
                    participant.species_id == species_id
                    for participant in reaction.reactants
                )
                or any(
                    participant.species_id == species_id
                    for participant in reaction.products
                )
            )
        ]


_REGISTRY: Optional[SBMLReactionRegistry] = None


def get_sbml_reaction_registry() -> SBMLReactionRegistry:
    """Return a lazily initialized reaction registry."""

    global _REGISTRY

    if _REGISTRY is None:
        _REGISTRY = SBMLReactionRegistry()

    return _REGISTRY


if __name__ == "__main__":
    registry = get_sbml_reaction_registry()

    print(
        f"Source: {registry.omex_path} :: {registry.sbml_filename}"
    )
    print(f"Reactions: {len(registry)}")

    for reaction in registry.list_reactions():
        reactants = " + ".join(
            f"{p.stoichiometry:g} {p.species_name}"
            for p in reaction.reactants
        ) or "∅"

        products = " + ".join(
            f"{p.stoichiometry:g} {p.species_name}"
            for p in reaction.products
        ) or "∅"

        print(f"\n[{reaction.id}] {reaction.name}")
        print(f"  {reactants} -> {products}")
        print(
            "  Direct parameter references: "
            + (
                ", ".join(reaction.direct_parameter_refs)
                or "none detected"
            )
        )

        if reaction.kinetic_law:
            print(f"  Rate law: {reaction.kinetic_law}")
            