"""Read parameter metadata from the exact SBML entry used by RoblitzModelRunner."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Optional
import zipfile

import libsbml

from src.HPO.model_runner import OMEX_PATH, SBML_FILENAME


EXPECTED_PARAMETER_COUNT = 141

CORE_CANDIDATES: dict[str, tuple[str, float]] = {
    "p1": ("b_syn_LH", 7309.91587614104),
    "p17": ("b_rel_FSH", 0.0569894397708967),
    "p49": ("k_AF1", 3.66180418829425),
    "p51": ("k_AF2_AF3", 4.88231609092536),
    "p173": ("b_IhB", 89.4934669769107),
}


@dataclass(frozen=True)
class SBMLParameter:
    """Metadata for one global SBML parameter."""

    id: str
    name: str
    value: float
    units: str
    constant: bool


class SBMLParameterRegistry:
    """Read-only registry for global parameters in the active Röblitz model."""

    def __init__(
        self,
        omex_path: Path | str = OMEX_PATH,
        sbml_filename: str = SBML_FILENAME,
        expected_count: Optional[int] = EXPECTED_PARAMETER_COUNT,
    ) -> None:
        self.omex_path = Path(omex_path)
        self.sbml_filename = sbml_filename

        self._by_id: dict[str, SBMLParameter] = {}
        self._by_name: dict[str, SBMLParameter] = {}
        self._parameters: list[SBMLParameter] = []

        self._load(expected_count)

    def _load(self, expected_count: Optional[int]) -> None:
        """Extract and parse the same SBML file used by the model runner."""

        if not self.omex_path.is_file():
            raise FileNotFoundError(
                f"Röblitz OMEX file not found: {self.omex_path}"
            )

        try:
            with zipfile.ZipFile(self.omex_path) as archive:
                if self.sbml_filename not in archive.namelist():
                    raise FileNotFoundError(
                        f"SBML entry {self.sbml_filename!r} "
                        f"not found inside {self.omex_path}"
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
            raise RuntimeError(
                "The SBML document contains no model."
            )

        count = model.getNumParameters()

        if expected_count is not None and count != expected_count:
            raise ValueError(
                f"Expected {expected_count} global parameters, "
                f"found {count}."
            )

        for index in range(count):
            param = model.getParameter(index)

            param_id = param.getId()
            param_name = param.getName() or param_id
            value = float(param.getValue())

            if not param_id:
                raise ValueError(
                    f"Global parameter at index {index} has no ID."
                )

            if not math.isfinite(value):
                raise ValueError(
                    f"Parameter {param_id!r} has "
                    f"non-finite value {value}."
                )

            if param_id in self._by_id:
                raise ValueError(
                    f"Duplicate SBML parameter ID {param_id!r}."
                )

            if param_name in self._by_name:
                raise ValueError(
                    f"Duplicate SBML parameter name {param_name!r}; "
                    "lookup is ambiguous."
                )

            item = SBMLParameter(
                id=param_id,
                name=param_name,
                value=value,
                units=param.getUnits() or "",
                constant=bool(param.getConstant()),
            )

            self._by_id[item.id] = item
            self._by_name[item.name] = item
            self._parameters.append(item)

    def __len__(self) -> int:
        return len(self._parameters)

    def __contains__(self, key: str) -> bool:
        return key in self._by_id or key in self._by_name

    def __getitem__(self, key: str) -> SBMLParameter:
        return self.get(key)

    def get(self, key: str) -> SBMLParameter:
        """Retrieve a parameter using its SBML ID or name."""

        if key in self._by_id:
            return self._by_id[key]

        if key in self._by_name:
            return self._by_name[key]

        raise KeyError(
            f"Unknown Röblitz SBML parameter: {key!r}"
        )

    def resolve_id(self, key: str) -> str:
        """Resolve an ID or name to its SBML ID."""
        return self.get(key).id

    def resolve_name(self, key: str) -> str:
        """Resolve an ID or name to its SBML name."""
        return self.get(key).name

    def get_baseline_value(self, key: str) -> float:
        """Get the original SBML value by ID or name."""
        return self.get(key).value

    def list_parameters(self) -> list[SBMLParameter]:
        """Return all parameters in their SBML definition order."""
        return list(self._parameters)

    def to_dict(self, by: str = "id") -> dict[str, float]:
        """Export parameter values indexed by ID or name."""

        if by == "id":
            return {
                param.id: param.value
                for param in self._parameters
            }

        if by == "name":
            return {
                param.name: param.value
                for param in self._parameters
            }

        raise ValueError(
            "by must be either 'id' or 'name'"
        )

    def verify_core_candidates(self) -> dict[str, bool]:
        """Check the expected IDs, names, and baseline values of five candidates."""

        results = {}

        for param_id, (
            expected_name,
            expected_value,
        ) in CORE_CANDIDATES.items():

            param = self.get(param_id)

            results[param_id] = (
                param.name == expected_name
                and math.isclose(
                    param.value,
                    expected_value,
                    rel_tol=1e-5,
                    abs_tol=1e-12,
                )
            )

        return results

    def export_csv(self, output_path: Path | str) -> Path:
        """Export the parameter registry to a CSV file."""

        import csv

        path = Path(output_path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as stream:

            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "id",
                    "name",
                    "value",
                    "units",
                    "constant",
                ],
            )

            writer.writeheader()

            for param in self._parameters:
                writer.writerow({
                    "id": param.id,
                    "name": param.name,
                    "value": param.value,
                    "units": param.units,
                    "constant": param.constant,
                })

        return path


_REGISTRY: Optional[SBMLParameterRegistry] = None


def get_sbml_parameter_registry() -> SBMLParameterRegistry:
    """Return a lazily initialized registry for the active OMEX model."""

    global _REGISTRY

    if _REGISTRY is None:
        _REGISTRY = SBMLParameterRegistry()

    return _REGISTRY


if __name__ == "__main__":
    registry = get_sbml_parameter_registry()

    print(f"Source: {registry.omex_path} :: {registry.sbml_filename}")
    print(f"Global parameters: {len(registry)}")

    for param_id, passed in registry.verify_core_candidates().items():
        param = registry.get(param_id)

        status = "PASS" if passed else "FAIL"

        print(
            f"{status} {param_id:5s} "
            f"{param.name:14s} {param.value:.12g}"
        )