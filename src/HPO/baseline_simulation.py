from dataclasses import dataclass
from typing import Any, Optional, Dict

from src.twin.simulation_guard import safe_simulate, SimulationResult


@dataclass
class BaselineSimulation:
    patient_id: int
    start_day: float
    end_day: float
    points: int
    result: Any
    parameter_overrides: Optional[Dict[str, float]] = None
    simulation_result: Optional[SimulationResult] = None


def run_baseline_simulation(
    patient_id: int,
    start_day: float = 0,
    end_day: float = 120,
    points: int = 1201,
    parameter_overrides: Optional[Dict[str, float]] = None,
) -> BaselineSimulation:
    sim_res = safe_simulate(
        parameter_overrides=parameter_overrides,
        start_day=start_day,
        end_day=end_day,
        points=points,
    )

    if not sim_res.success:
        raise RuntimeError(
            f"Simulation failed for patient {patient_id}: {sim_res.error_message}"
        )

    return BaselineSimulation(
        patient_id=patient_id,
        start_day=start_day,
        end_day=end_day,
        points=points,
        result=sim_res.trajectory,
        parameter_overrides=parameter_overrides,
        simulation_result=sim_res,
    )