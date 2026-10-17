"""Run controlled what-if scenarios using the Röblitz HPO model."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from src.HPO.model_runner import RoblitzModelRunner
from src.HPO.sbml_parameters import SBMLParameterRegistry


@dataclass
class TwinScenario:
    """Results and parameter overrides for one what-if scenario."""

    name: str
    parameter_changes: dict[str, float]
    result: Any = None


class ScenarioSimulator:
    """
    Runs independent what-if simulations using the Röblitz model.

    Parameter changes represent NEW parameter values, not increments.

    Both SBML IDs (for example, p1) and SBML parameter names
    (for example, b_syn_LH) are supported.
    """

    def __init__(
        self,
        runner: RoblitzModelRunner | None = None,
    ):
        self.runner = runner or RoblitzModelRunner()

        self.parameter_registry = SBMLParameterRegistry(
            omex_path=self.runner.omex_path,
        )

    def _resolve_parameter_id(
        self,
        model,
        parameter_name: str,
    ) -> str:
        """Resolve an SBML ID or parameter name to its SBML ID."""

        try:
            parameter_id = self.parameter_registry.resolve_id(
                parameter_name
            )
        except KeyError as exc:
            raise ValueError(
                f"Unknown Röblitz model parameter: {parameter_name!r}"
            ) from exc

        model_parameter_ids = set(
            model.getGlobalParameterIds()
        )

        if parameter_id not in model_parameter_ids:
            raise ValueError(
                f"Parameter {parameter_id!r} exists in the registry "
                "but not in the active simulation model."
            )

        return parameter_id

    @staticmethod
    def _validate_time_range(
        start_day: float,
        end_day: float,
        points: int,
    ) -> None:
        """Validate requested simulation boundaries."""

        if not math.isfinite(start_day):
            raise ValueError("start_day must be finite.")

        if not math.isfinite(end_day):
            raise ValueError("end_day must be finite.")

        if end_day <= start_day:
            raise ValueError(
                "end_day must be greater than start_day."
            )

        if isinstance(points, bool) or not isinstance(points, int):
            raise ValueError("points must be an integer.")

        if points < 2:
            raise ValueError("points must be at least 2.")

    def run(
        self,
        name: str,
        parameter_changes: dict[str, float],
        start_day: float = 0,
        end_day: float = 120,
        points: int = 1201,
    ) -> TwinScenario:
        """
        Run a what-if scenario on a completely fresh model.

        Example:
            {"b_syn_LH": 8000.0}

        This sets the parameter to 8000; it does not add 8000
        to the original parameter value.
        """

        if not isinstance(name, str) or not name.strip():
            raise ValueError("Scenario name must be a non-empty string.")

        if not isinstance(parameter_changes, dict):
            raise TypeError(
                "parameter_changes must be a dictionary."
            )

        self._validate_time_range(
            start_day,
            end_day,
            points,
        )

        if not parameter_changes:
            raise ValueError(
                "Provide at least one parameter override."
            )

        # Resolve and validate inputs before modifying the model.
        resolved_changes: dict[str, float] = {}

        for parameter_name, raw_value in parameter_changes.items():
            if not isinstance(parameter_name, str):
                raise TypeError(
                    "Parameter names must be strings."
                )

            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid value for {parameter_name!r}: "
                    f"{raw_value!r}"
                ) from exc

            if not math.isfinite(value):
                raise ValueError(
                    f"Parameter {parameter_name!r} must have "
                    "a finite value."
                )

            parameter_id = self._resolve_parameter_id(
                self.runner.model,
                parameter_name,
            )

            if parameter_id in resolved_changes:
                raise ValueError(
                    f"Parameter {parameter_id!r} was supplied "
                    "more than once through different names."
                )

            resolved_changes[parameter_id] = value

        # Every scenario receives its own independent model.
        model = self.runner.create_fresh_model()

        for parameter_id, value in resolved_changes.items():
            model[parameter_id] = value

        result = model.simulate(
            start_day,
            end_day,
            points,
        )

        output = np.asarray(result, dtype=float)

        if output.ndim != 2 or output.shape[0] == 0:
            raise RuntimeError(
                f"Scenario {name!r} returned an invalid output shape: "
                f"{output.shape}."
            )

        if not np.all(np.isfinite(output)):
            raise RuntimeError(
                f"Scenario {name!r} produced non-finite output values."
            )

        # Store canonical SBML parameter IDs and the actual assigned
        # values so the scenario is unambiguous and reproducible.
        return TwinScenario(
            name=name,
            parameter_changes=dict(resolved_changes),
            result=result,
        )