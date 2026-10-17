"""Phase 7.1 tests for the active Röblitz SBML parameter registry."""

import math
import unittest

from src.HPO.model_runner import RoblitzModelRunner

from src.HPO.sbml_parameters import (
    CORE_CANDIDATES,
    get_sbml_parameter_registry,
)


class TestSBMLParameterRegistry(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.registry = get_sbml_parameter_registry()

    def test_expected_parameter_count(self):
        self.assertEqual(
            len(self.registry),
            141,
        )

    def test_ids_and_names_are_unique(self):
        params = self.registry.list_parameters()

        ids = {param.id for param in params}
        names = {param.name for param in params}

        self.assertEqual(len(ids), len(params))
        self.assertEqual(len(names), len(params))

    def test_values_are_finite(self):
        self.assertTrue(
            all(
                math.isfinite(param.value)
                for param in self.registry.list_parameters()
            )
        )

    def test_id_name_lookup(self):
        self.assertEqual(
            self.registry.get("p1"),
            self.registry.get("b_syn_LH"),
        )

        self.assertEqual(
            self.registry.resolve_id("b_syn_LH"),
            "p1",
        )

        self.assertEqual(
            self.registry.resolve_name("p1"),
            "b_syn_LH",
        )

    def test_core_candidate_ids_names_and_values(self):
        results = self.registry.verify_core_candidates()

        self.assertEqual(
            set(results),
            set(CORE_CANDIDATES),
        )

        self.assertTrue(
            all(results.values()),
            msg=f"Core candidate verification failed: {results}",
        )

    def test_same_parameters_as_roadrunner_model(self):
        """
        Ensure the registry inspects the SBML used by the simulation runner.
        """

        runner = RoblitzModelRunner()

        runner_ids = set(runner.get_parameters())

        registry_ids = {
            param.id
            for param in self.registry.list_parameters()
        }

        self.assertEqual(
            registry_ids,
            runner_ids,
        )

        for param_id in registry_ids:
            registry_value = (
                self.registry.get_baseline_value(param_id)
            )

            runner_value = float(
                runner.model[param_id]
            )

            self.assertTrue(
                math.isclose(
                    registry_value,
                    runner_value,
                    rel_tol=1e-8,
                    abs_tol=1e-12,
                ),
                msg=(
                    f"SBML registry and RoadRunner disagree "
                    f"for parameter {param_id}: "
                    f"{registry_value} != {runner_value}"
                ),
            )

    def test_unknown_parameter_raises(self):
        with self.assertRaises(KeyError):
            self.registry.get("not_a_real_parameter")


if __name__ == "__main__":
    unittest.main(verbosity=2)