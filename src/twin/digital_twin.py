"""Patient digital twin orchestration and scenario comparisons."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.HPO.baseline_simulation import run_baseline_simulation
from src.HPO.patient_values import HPOPatientValues
from src.twin.parameter_mapping import PatientTwinMapping
from src.twin.personalization import (
    PatientProfile,
    create_patient_profile,
)
from src.twin.scenarios import ScenarioSimulator


@dataclass
class DigitalTwinState:
    """Stored baseline and what-if results."""

    baseline: Any = None
    scenarios: dict[str, Any] = field(default_factory=dict)


class PatientDigitalTwin:
    """
    Organizes patient observations, baseline simulation, and scenarios.

    Clinical observations are not automatically written into model
    kinetic parameters. Patient-specific calibration must be implemented
    and validated separately.
    """

    def __init__(self, patient: HPOPatientValues):
        self.patient = patient
        self.scenario_simulator = ScenarioSimulator()

        self.profile: PatientProfile = create_patient_profile(patient)
        self.mapping: PatientTwinMapping = self.profile.mapping

        self.state = DigitalTwinState()
        self.ml_assessment = None
        self.status = "initialized"

    @staticmethod
    def _column_names(result) -> list[str]:
        """Read column labels from a RoadRunner simulation result."""
        names = result.colnames

        if callable(names):
            names = names()

        return [str(name) for name in names]

    def run_baseline(
        self,
        start_day: float = 0,
        end_day: float = 120,
        points: int = 1201,
    ):
        """
        Run the original Röblitz model's baseline parameterization.

        This includes the scheduled events defined in the source model.
        It is not automatically an untreated or patient-calibrated baseline.
        """
        simulation = run_baseline_simulation(
            patient_id=self.patient.patient_id,
            start_day=start_day,
            end_day=end_day,
            points=points,
        )

        self.state.baseline = simulation
        self.status = "baseline_ready"

        return simulation

    def run_scenario(
        self,
        name: str,
        parameter_changes: dict[str, float],
        start_day: float = 0,
        end_day: float = 120,
        points: int = 1201,
    ):
        """
        Run and store a what-if scenario.

        Parameter changes are replacement values, not additive changes.
        For example, {"p1": 8000.0} sets p1 to 8000.
        """
        scenario = self.scenario_simulator.run(
            name=name,
            parameter_changes=parameter_changes,
            start_day=start_day,
            end_day=end_day,
            points=points,
        )

        self.state.scenarios[name] = scenario
        self.status = "scenario_ready"

        return scenario

    def compare_baseline_scenario(
        self,
        scenario_name: str,
    ) -> dict[str, dict[str, float | None]]:
        """Compare results only when variables and time grids align."""
        if self.state.baseline is None:
            raise RuntimeError(
                "Run the baseline before comparing scenarios."
            )

        if scenario_name not in self.state.scenarios:
            raise ValueError(
                f"Scenario {scenario_name!r} does not exist."
            )

        baseline_result = self.state.baseline.result
        scenario_result = self.state.scenarios[scenario_name].result

        baseline_columns = self._column_names(baseline_result)
        scenario_columns = self._column_names(scenario_result)

        if set(baseline_columns) != set(scenario_columns):
            raise ValueError(
                "Baseline and scenario contain different model variables."
            )

        baseline_array = np.asarray(baseline_result, dtype=float)
        scenario_array = np.asarray(scenario_result, dtype=float)

        if baseline_array.shape != scenario_array.shape:
            raise ValueError(
                "Baseline and scenario trajectories have different shapes. "
                "Run both with the same boundaries and point count."
            )

        if "time" not in baseline_columns:
            raise ValueError(
                "Simulation output does not contain a time column."
            )

        baseline_time_index = baseline_columns.index("time")
        scenario_time_index = scenario_columns.index("time")

        baseline_times = baseline_array[:, baseline_time_index]
        scenario_times = scenario_array[:, scenario_time_index]

        if not np.allclose(
            baseline_times,
            scenario_times,
            rtol=0,
            atol=1e-9,
        ):
            raise ValueError(
                "Baseline and scenario time points do not align."
            )

        if not np.all(np.isfinite(baseline_array)):
            raise ValueError("Baseline contains non-finite values.")

        if not np.all(np.isfinite(scenario_array)):
            raise ValueError("Scenario contains non-finite values.")

        scenario_index = {
            column: index
            for index, column in enumerate(scenario_columns)
        }

        comparison: dict[str, dict[str, float | None]] = {}

        for column in baseline_columns:
            if column == "time":
                continue

            baseline_values = baseline_array[
                :, baseline_columns.index(column)
            ]
            scenario_values = scenario_array[
                :, scenario_index[column]
            ]

            difference = scenario_values - baseline_values
            baseline_final = float(baseline_values[-1])
            scenario_final = float(scenario_values[-1])

            if abs(baseline_final) > 1e-12:
                relative_change_pct = (
                    (scenario_final - baseline_final)
                    / abs(baseline_final)
                    * 100.0
                )
            else:
                relative_change_pct = None

            comparison[column] = {
                "baseline_final": baseline_final,
                "scenario_final": scenario_final,
                "final_difference": float(difference[-1]),
                "relative_final_change_pct": relative_change_pct,
                "maximum_absolute_difference": float(
                    np.max(np.abs(difference))
                ),
                "mean_absolute_difference": float(
                    np.mean(np.abs(difference))
                ),
                "rmse": float(
                    np.sqrt(np.mean(difference ** 2))
                ),
            }

        return comparison

    def attach_ml_assessment(self, assessment) -> None:
        """Attach an ML assessment without modifying model parameters."""
        self.ml_assessment = assessment
        self.status = "ml_assessed"

    def add_scenario(
        self,
        name: str,
        simulation: Any,
    ) -> None:
        """Store an externally created simulation result."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                "Scenario name must be a non-empty string."
            )

        self.state.scenarios[name] = simulation

    def get_status(self) -> str:
        """Return the current lifecycle status."""
        return self.status

    def summary(self) -> dict:
        """Summarize the twin and clinical observation readiness."""
        return {
            "patient_id": self.patient.patient_id,
            "status": self.status,
            "baseline_available": self.state.baseline is not None,
            "scenario_count": len(self.state.scenarios),
            "scenario_names": list(self.state.scenarios.keys()),
            "ml_assessment_available": self.ml_assessment is not None,
            "clinical_profile": self.profile.summary(),
        }


def _find_clinical_dataset(data_path=None):
    """Load the requested clinical CSV or prefer the preprocessed cohort."""
    import pandas as pd

    from src.Clinical.clinical_loader import (
        load_clinical_data,
        validate_dataset,
    )

    project_root = Path(__file__).resolve().parents[2]

    def load_and_validate(path):
        df = load_clinical_data(path)

        # The preprocessed file preserves the old source ratio in an
        # audit column and removes the corrupted FSH/LH source column.
        # Recreate a temporary schema-compatible alias for validation.
        if (
            "FSH/LH" not in df.columns
            and "FSH_LH_source_raw" in df.columns
        ):
            df["FSH/LH"] = df["FSH_LH_source_raw"]

        validate_dataset(df)

        if "Patient File No." not in df.columns:
            raise ValueError(
                "Dataset is missing the required 'Patient File No.' column."
            )

        return path, df

    if data_path:
        path = Path(data_path)

        if not path.is_absolute():
            path = project_root / path

        if not path.is_file():
            raise FileNotFoundError(
                f"Clinical dataset not found: {path}"
            )

        return load_and_validate(path)

    # Use recalculated features when the preprocessed file exists.
    preferred_paths = [
        project_root / "data" / "clinical_dataset_preprocessed.csv",
        project_root / "data" / "PCOS_data_without_infertility-Full_new.csv",
    ]

    for preferred_path in preferred_paths:
        if preferred_path.is_file():
            return load_and_validate(preferred_path)

    ignored_directories = {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        "node_modules",
        "models",
        "src",
    }

    candidates = []

    for path in project_root.rglob("*.csv"):
        if any(
            part.lower() in ignored_directories
            for part in path.relative_to(project_root).parts
        ):
            continue

        try:
            _, df = load_and_validate(path)
        except Exception:
            continue

        ids = pd.to_numeric(
            df["Patient File No."],
            errors="coerce",
        )

        if ids.notna().any():
            candidates.append((path, df))

    if not candidates:
        raise FileNotFoundError(
            "No compatible clinical CSV was found in the project.\n"
            "Place the clinical dataset in the project, or pass "
            "--data path\\to\\clinical_dataset.csv."
        )

    cohort_candidates = [
        item for item in candidates if len(item[1]) == 541
    ]

    if len(cohort_candidates) == 1:
        return cohort_candidates[0]

    if len(candidates) == 1:
        return candidates[0]

    named_candidates = [
        item for item in candidates
        if "clinical" in item[0].name.lower()
    ]

    if len(named_candidates) == 1:
        return named_candidates[0]

    paths = "\n".join(
        f"  - {path.relative_to(project_root)} ({len(df)} rows)"
        for path, df in candidates
    )

    raise RuntimeError(
        "Multiple compatible clinical datasets were found. "
        "Choose the correct file using --data:\n"
        + paths
    )


def main():
    """Run the patient-to-scenario digital twin workflow."""
    import argparse
    import math

    import pandas as pd

    from src.Clinical.hpo_observation_builder import (
        build_hpo_patient_values,
    )
    from src.HPO.sbml_parameters import (
        get_sbml_parameter_registry,
    )

    parser = argparse.ArgumentParser(
        description="Run the OvaTwin clinical-to-what-if workflow."
    )

    parser.add_argument(
        "--data",
        default=None,
        help="Path to the clinical CSV.",
    )
    parser.add_argument(
        "--patient-id",
        default=None,
        help="Patient File No.; defaults to the first valid row.",
    )
    parser.add_argument("--start-day", type=float, default=0.0)
    parser.add_argument("--end-day", type=float, default=120.0)
    parser.add_argument("--points", type=int, default=1201)
    parser.add_argument(
        "--parameter",
        default="p1",
        help="SBML parameter ID or name for the sensitivity scenario.",
    )
    parser.add_argument(
        "--change-percent",
        type=float,
        default=1.0,
        help="Percentage change to the parameter's original value.",
    )

    args = parser.parse_args()

    if (
        not math.isfinite(args.start_day)
        or not math.isfinite(args.end_day)
        or args.end_day <= args.start_day
    ):
        parser.error(
            "--end-day must be finite and greater than --start-day."
        )

    if args.points < 2:
        parser.error("--points must be at least 2.")

    if not math.isfinite(args.change_percent):
        parser.error("--change-percent must be finite.")

    print("=" * 68)
    print("OVATWIN — COMPLETE WORKFLOW")
    print("=" * 68)

    # 1. Load the clinical cohort.
    dataset_path, df = _find_clinical_dataset(args.data)

    print(f"\nDataset: {dataset_path}")
    print(f"Dataset rows: {len(df)}")

    if "Patient File No." not in df.columns:
        raise ValueError("Patient File No. column is required.")

    patient_ids = pd.to_numeric(
        df["Patient File No."],
        errors="coerce",
    )

    valid_mask = (
        patient_ids.notna()
        & np.isfinite(patient_ids)
        & patient_ids.gt(0)
        & patient_ids.mod(1).eq(0)
    )

    df = df.loc[valid_mask].copy()
    patient_ids = patient_ids.loc[valid_mask]

    if df.empty:
        raise RuntimeError(
            "No rows have a valid positive integer patient ID."
        )

    # 2. Select a patient by ID or default to the first valid row.
    if args.patient_id is None:
        selected_index = df.index[0]
    else:
        try:
            requested_id = float(args.patient_id)
        except ValueError:
            matches = (
                df["Patient File No."].astype(str).str.strip()
                == args.patient_id.strip()
            )
        else:
            matches = patient_ids == requested_id

        matching_indices = df.index[matches]

        if len(matching_indices) == 0:
            raise ValueError(
                f"Patient ID {args.patient_id!r} was not found."
            )

        if len(matching_indices) > 1:
            raise ValueError(
                f"Patient ID {args.patient_id!r} is not unique."
            )

        selected_index = matching_indices[0]

    row = df.loc[selected_index]

    # 3. Convert the selected clinical row into observations.
    patient = build_hpo_patient_values(row)

    print("\n=== SELECTED PATIENT ===")
    print(f"Patient ID: {patient.patient_id}")
    print(f"Age: {patient.age}")
    print(f"Weight: {patient.weight}")
    print(f"Height: {patient.height}")
    print(f"BMI observation: {patient.bmi}")
    print(f"Waist/hip ratio observation: {patient.waist_hip_ratio}")
    print(f"LH observation: {patient.lh}")
    print(f"FSH observation: {patient.fsh}")
    print(f"FSH/LH ratio observation: {patient.fsh_lh_ratio}")
    print(f"AMH observation: {patient.amh}")
    print(
        "Cycle status code (not interpreted): "
        f"{patient.cycle_status_encoded}"
    )

    if "PCOS (Y/N)" in row.index:
        print(f"Dataset PCOS label: {row.get('PCOS (Y/N)')}")

    # 4. Create the patient twin and report observation readiness.
    twin = PatientDigitalTwin(patient)

    print("\n=== PATIENT PROFILE ===")
    print(twin.profile.observation_summary())

    if twin.profile.data_quality_flags:
        print("\n=== PATIENT DATA QUALITY FLAGS ===")
        for flag in twin.profile.data_quality_flags:
            print(f"- {flag}")
    else:
        print("\nNo configured data-quality flags raised.")

    # 5. Run the original-model baseline.
    print("\n=== BASELINE SIMULATION ===")

    baseline = twin.run_baseline(
        start_day=args.start_day,
        end_day=args.end_day,
        points=args.points,
    )

    baseline_values = np.asarray(baseline.result, dtype=float)

    print(f"Output shape: {baseline_values.shape}")
    print(
        "All output values finite: "
        f"{np.isfinite(baseline_values).all()}"
    )

    # 6. Configure a parameter sensitivity scenario.
    registry = get_sbml_parameter_registry()

    parameter_id = registry.resolve_id(args.parameter)
    original_value = registry.get_baseline_value(parameter_id)

    scenario_value = (
        original_value
        * (1.0 + args.change_percent / 100.0)
    )

    if not math.isfinite(scenario_value):
        raise ValueError(
            "Calculated scenario parameter value is not finite."
        )

    if args.change_percent != 0 and original_value == 0:
        raise ValueError(
            "A percentage change cannot modify a zero-valued parameter. "
            "Choose another parameter or implement an absolute-value "
            "scenario separately."
        )

    scenario_name = (
        f"{parameter_id}_{args.change_percent:+g}pct_sensitivity"
    )

    print("\n=== WHAT-IF SCENARIO ===")
    print(f"Scenario: {scenario_name}")
    print(
        f"Parameter: {parameter_id} "
        f"({registry.get(parameter_id).name})"
    )
    print(f"Original value: {original_value:.12g}")
    print(f"Scenario value: {scenario_value:.12g}")

    scenario = twin.run_scenario(
        name=scenario_name,
        parameter_changes={parameter_id: scenario_value},
        start_day=args.start_day,
        end_day=args.end_day,
        points=args.points,
    )

    scenario_values = np.asarray(scenario.result, dtype=float)

    print(f"Output shape: {scenario_values.shape}")
    print(
        "All output values finite: "
        f"{np.isfinite(scenario_values).all()}"
    )

    # 7. Compare baseline and what-if results.
    print("\n=== BASELINE VS WHAT-IF ===")

    comparison = twin.compare_baseline_scenario(scenario_name)

    target_species = {
        "[LH_bld]",
        "[FSH_bld]",
        "[E2]",
        "[P4]",
    }

    for variable, metrics in comparison.items():
        normalized = variable.strip("[]")

        if f"[{normalized}]" not in target_species:
            continue

        print(f"\n{normalized}")

        for metric_name, value in metrics.items():
            if value is None:
                print(f"  {metric_name}: not calculable")
            else:
                print(f"  {metric_name}: {value:.8g}")

    # 8. Summarize status.
    print("\n=== DIGITAL TWIN SUMMARY ===")
    print(twin.summary())

    print("\n" + "=" * 68)
    print("WORKFLOW COMPLETED")
    print("=" * 68)
    print(
        "The scenario is a model-parameter sensitivity experiment. "
        "It is not a clinically validated patient-specific prediction."
    )


if __name__ == "__main__":
    main()