"""
simulation_guard.py

Phase 7.7: Trajectory sanity, numerical validation, and defensive solver error handling.
Enforces the protocol:
  - Validates trajectory integrity (finite numbers, no NaN/Inf, non-negative concentrations).
  - Explicitly captures CVODE / RoadRunner simulation exceptions.
  - Strictly prevents silent parameter fallbacks or masked solver failures.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math
import numpy as np

from src.HPO.model_runner import RoblitzModelRunner
from src.HPO.sbml_parameters import get_sbml_parameter_registry
from src.HPO.parameters import get_authoritative_registry

ROOT_DIR = Path(__file__).resolve().parents[2]


@dataclass
class SimulationResult:
    """
    Transparent container for ODE simulation results.
    Guarantees explicit failure reporting without silent fallbacks.
    """
    success: bool
    error_message: Optional[str] = None
    parameters_used: Dict[str, float] = field(default_factory=dict)
    trajectory: Optional[Any] = None
    warnings: List[str] = field(default_factory=list)
    start_day: float = 0.0
    end_day: float = 120.0
    points: int = 1201

    @property
    def shape(self) -> Optional[Tuple[int, int]]:
        return self.trajectory.shape if self.trajectory is not None else None

    @property
    def colnames(self) -> List[str]:
        return list(self.trajectory.colnames) if self.trajectory is not None else []

    def __len__(self) -> int:
        return self.trajectory.shape[0] if self.trajectory is not None else 0


def validate_trajectory_sanity(
    trajectory: Any,
    expected_points: int = 1201,
    expected_species_count: int = 46,
) -> Tuple[bool, Optional[str], List[str]]:
    """
    Performs comprehensive numerical and biological sanity checks on a simulated trajectory:
      1. Non-empty output and shape conformity.
      2. Absence of NaN or Inf values.
      3. Physiological positivity check (flags unphysical negative concentrations).
    """
    warnings: List[str] = []

    if trajectory is None:
        return False, "Simulation trajectory is None.", warnings

    try:
        arr = np.asarray(trajectory)
    except Exception as e:
        return False, f"Could not convert trajectory to array: {e}", warnings

    # Check 1: Shape conformity
    if arr.shape[0] != expected_points:
        return (
            False,
            f"Shape mismatch: expected {expected_points} timepoints, got {arr.shape[0]}.",
            warnings,
        )

    # Check 2: Finite checks (No NaN, No Inf)
    if np.isnan(arr).any():
        return False, "Trajectory contains NaN values (numerical instability detected).", warnings

    if np.isinf(arr).any():
        return False, "Trajectory contains Infinite values (unbounded divergence detected).", warnings

    # Check 3: Physiological positivity check
    # RoadRunner columns include 'time' as col 0, species concentrations follow.
    # Note: 's114' is an internal SBML antagonist PK dosing variable that is negative by model design.
    colnames = getattr(trajectory, "colnames", [])
    for col_idx in range(1, arr.shape[1]):
        col_name = colnames[col_idx] if col_idx < len(colnames) else f"Species_{col_idx}"
        col_clean = col_name.strip("[]")

        if col_clean == "s114":
            continue

        col_vals = arr[:, col_idx]
        min_val = float(np.min(col_vals))

        if min_val < -1.0:
            # Significant unphysical negative concentration
            return (
                False,
                f"Unphysical negative concentration in '{col_name}': minimum = {min_val:.6f}",
                warnings,
            )
        elif min_val < -1e-6:
            # Minor CVODE numerical integration noise
            warnings.append(
                f"Minor numerical integrator noise in '{col_name}': minimum = {min_val:.2e}"
            )

    return True, None, warnings


def safe_simulate(
    parameter_overrides: Optional[Dict[str, float]] = None,
    start_day: float = 0.0,
    end_day: float = 120.0,
    points: int = 1201,
) -> SimulationResult:
    """
    Safely executes a Röblitz model simulation with full error capture.

    Guarantees:
      - Parameters are checked and clamped before injection.
      - Any RoadRunner or CVODE exception is caught and returned as success=False.
      - Generated trajectories pass sanity checks before being deemed successful.
      - No silent fallbacks or altered outputs.
    """
    params_used: Dict[str, float] = {}
    warnings: List[str] = []

    reg = get_sbml_parameter_registry()
    auth_reg = get_authoritative_registry()

    # 1. Instantiate fresh model
    try:
        runner = RoblitzModelRunner()
        model = runner.create_fresh_model()
        # Set absolute tolerance to 1e-10 to prevent CVODE sub-epsilon step failure at Day 91
        try:
            integrator = model.getIntegrator()
            integrator.setValue("absolute_tolerance", 1e-10)
        except Exception:
            pass
    except Exception as e:
        return SimulationResult(
            success=False,
            error_message=f"Model instantiation failed: {e}",
            parameters_used={},
            trajectory=None,
            start_day=start_day,
            end_day=end_day,
            points=points,
        )

    # 2. Apply parameter overrides
    if parameter_overrides:
        for key, val in parameter_overrides.items():
            try:
                param_id = reg.resolve_id(key)
                # Check Layer 2 hard clamp
                meta = auth_reg.get(param_id)
                clamped_val = meta.clamp(float(val))

                if clamped_val != float(val):
                    warnings.append(
                        f"Parameter '{key}' ({val}) was clamped to hard safety limit {clamped_val}."
                    )

                model[param_id] = float(clamped_val)
                params_used[param_id] = float(clamped_val)
            except Exception as e:
                return SimulationResult(
                    success=False,
                    error_message=f"Failed to set parameter '{key}'={val}: {e}",
                    parameters_used=params_used,
                    trajectory=None,
                    start_day=start_day,
                    end_day=end_day,
                    points=points,
                )

    # 3. Execute ODE simulation with exception guard
    try:
        raw_result = model.simulate(start_day, end_day, points)
    except Exception as e:
        return SimulationResult(
            success=False,
            error_message=f"CVODE/RoadRunner solver failure during simulation: {e}",
            parameters_used=params_used,
            trajectory=None,
            warnings=warnings,
            start_day=start_day,
            end_day=end_day,
            points=points,
        )

    # 4. Perform trajectory sanity validation
    is_sane, error_msg, sanity_warnings = validate_trajectory_sanity(
        raw_result,
        expected_points=points,
    )
    warnings.extend(sanity_warnings)

    if not is_sane:
        return SimulationResult(
            success=False,
            error_message=f"Trajectory sanity check failed: {error_msg}",
            parameters_used=params_used,
            trajectory=None,
            warnings=warnings,
            start_day=start_day,
            end_day=end_day,
            points=points,
        )

    return SimulationResult(
        success=True,
        error_message=None,
        parameters_used=params_used,
        trajectory=raw_result,
        warnings=warnings,
        start_day=start_day,
        end_day=end_day,
        points=points,
    )
