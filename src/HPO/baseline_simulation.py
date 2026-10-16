from dataclasses import dataclass
from typing import Any, Optional, Dict

from src.HPO.model_runner import RoblitzModelRunner
from src.HPO.sbml_parameters import get_sbml_parameter_registry


@dataclass
class BaselineSimulation:
    patient_id: int
    start_day: float
    end_day: float
    points: int
    result: Any
    parameter_overrides: Optional[Dict[str, float]] = None


def run_baseline_simulation(
    patient_id: int,
    start_day: float = 0,
    end_day: float = 120,
    points: int = 1201,
    parameter_overrides: Optional[Dict[str, float]] = None,
) -> BaselineSimulation:
    runner = RoblitzModelRunner()

    if parameter_overrides:
        model = runner.create_fresh_model()
        reg = get_sbml_parameter_registry()

        for k, v in parameter_overrides.items():
            param_id = reg.resolve_id(k)
            model[param_id] = float(v)

        result = model.simulate(start_day, end_day, points)
    else:
        result = runner.simulate(
            start=start_day,
            end=end_day,
            points=points,
        )

    return BaselineSimulation(
        patient_id=patient_id,
        start_day=start_day,
        end_day=end_day,
        points=points,
        result=result,
        parameter_overrides=parameter_overrides,
    )