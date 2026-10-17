"""Tests for the active Röblitz SBML reaction registry."""

import unittest

from src.HPO.model_runner import RoblitzModelRunner
from src.HPO.sbml_reactions import (
    EXPECTED_REACTION_COUNT,
    get_sbml_reaction_registry,
)


class TestSBMLReactionRegistry(unittest.TestCase):
    def test_known_parameter_references_are_detected(self):
        """Verify references visible in known reaction rate laws."""

        expected_reactions = {
            "p1": {"re25"},
            "p17": {"re29", "re84"},
            "p49": {"re43"},
            "p51": {"re44"},
            "p173": {"re72"},
        }

        for parameter_id, expected_ids in expected_reactions.items():
            with self.subTest(parameter=parameter_id):
                actual_ids = {
                    reaction.id
                    for reaction in self.registry.find_by_parameter(
                        parameter_id
                    )
                }

                self.assertTrue(
                    expected_ids.issubset(actual_ids),
                    msg=(
                        f"{parameter_id}: expected reactions "
                        f"{sorted(expected_ids)}, found "
                        f"{sorted(actual_ids)}"
                    ),
                )
    @classmethod
    def setUpClass(cls):
        cls.registry = get_sbml_reaction_registry()

    def test_expected_reaction_count(self):
        self.assertEqual(
            len(self.registry),
            EXPECTED_REACTION_COUNT,
        )

    def test_reaction_ids_are_unique(self):
        reactions = self.registry.list_reactions()
        ids = {reaction.id for reaction in reactions}

        self.assertEqual(len(ids), len(reactions))

    def test_reactant_and_product_species_exist(self):
        valid_species = self.registry.list_species_ids()

        for reaction in self.registry.list_reactions():
            for participant in (
                reaction.reactants + reaction.products
            ):
                self.assertIn(
                    participant.species_id,
                    valid_species,
                    msg=(
                        f"{participant.species_id} referenced by "
                        f"{reaction.id} is not in the model."
                    ),
                )

    def test_all_reactions_can_be_retrieved_by_id(self):
        for reaction in self.registry.list_reactions():
            self.assertEqual(
                self.registry.get(reaction.id),
                reaction,
            )

    def test_unknown_reaction_raises(self):
        with self.assertRaises(KeyError):
            self.registry.get("not_a_real_reaction")

    def test_parameter_search_returns_valid_reactions(self):
        matches = self.registry.find_by_parameter("p1")
        all_ids = {
            reaction.id
            for reaction in self.registry.list_reactions()
        }

        for reaction in matches:
            self.assertIn(reaction.id, all_ids)
            self.assertIn(
                "p1",
                reaction.direct_parameter_refs,
            )

    def test_species_search_returns_valid_reactions(self):
        species_ids = self.registry.list_species_ids()
        self.assertTrue(species_ids)

        species_id = next(iter(species_ids))
        matches = self.registry.find_by_species(species_id)

        for reaction in matches:
            participant_ids = {
                participant.species_id
                for participant in (
                    reaction.reactants + reaction.products
                )
            }

            self.assertIn(species_id, participant_ids)

    def test_reaction_ids_exist_in_simulation_model(self):
        runner = RoblitzModelRunner()

        runner_reaction_ids = set(
            runner.model.getReactionIds()
        )

        registry_reaction_ids = {
            reaction.id
            for reaction in self.registry.list_reactions()
        }

        self.assertEqual(
            registry_reaction_ids,
            runner_reaction_ids,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
