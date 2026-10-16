"""
test_sbml_parameters.py

Unit test and verification suite for Phase 7.1:
Verifying the Röblitz SBML parameter extraction, count, values, and bidirectional lookups.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.HPO.sbml_parameters import (
    SBMLParameterRegistry,
    get_sbml_parameter_registry,
)


def test_parameter_count(registry: SBMLParameterRegistry):
    assert len(registry) == 141, f"Expected 141 parameters, found {len(registry)}"
    print("[OK] Test 1: Exactly 141 parameters extracted from SBML.")


def test_unique_identifiers(registry: SBMLParameterRegistry):
    ids = [p.id for p in registry.list_parameters()]
    names = [p.name for p in registry.list_parameters()]
    assert len(set(ids)) == 141, f"Duplicate IDs detected! Found {len(set(ids))} unique."
    assert len(set(names)) == 141, f"Duplicate names detected! Found {len(set(names))} unique."
    print("[OK] Test 2: All 141 parameter IDs and biological names are strictly unique.")


def test_finite_values(registry: SBMLParameterRegistry):
    import math
    for p in registry.list_parameters():
        assert not math.isnan(p.value), f"NaN found in parameter {p.id}"
        assert not math.isinf(p.value), f"Inf found in parameter {p.id}"
        assert isinstance(p.value, float), f"Value not float in parameter {p.id}"
    print("[OK] Test 3: All 141 baseline parameter values are valid finite floating-point numbers.")


def test_core_5_candidates(registry: SBMLParameterRegistry):
    verification = registry.verify_core_candidates()
    for param_id, matches in verification.items():
        assert matches, f"Core candidate parameter {param_id} verification failed!"
        param = registry.get(param_id)
        print(f"     -> {param.id:5s} | {param.name:12s} = {param.value:<18.10f} [VERIFIED]")
    print("[OK] Test 4: All 5 Tier-1 Core candidate parameters verified against published model values.")


def test_bidirectional_lookup(registry: SBMLParameterRegistry):
    # Lookup by ID
    p1_by_id = registry.get("p1")
    # Lookup by Name
    p1_by_name = registry.get("b_syn_LH")
    assert p1_by_id == p1_by_name
    assert registry.resolve_id("b_syn_LH") == "p1"
    assert registry.resolve_name("p1") == "b_syn_LH"
    assert registry.get_baseline_value("b_syn_LH") == p1_by_id.value

    # Check 'in' operator
    assert "p17" in registry
    assert "b_rel_FSH" in registry
    assert "non_existent_key_xyz" not in registry

    print("[OK] Test 5: Bidirectional lookup (ID <-> Name) and resolution verified.")


def test_unknown_parameter_raises(registry: SBMLParameterRegistry):
    try:
        registry.get("invalid_parameter_name_123")
        assert False, "Expected KeyError was not raised for invalid parameter."
    except KeyError as e:
        assert "Unknown Röblitz parameter" in str(e)
    print("[OK] Test 6: Unknown parameter query raises explicit KeyError with guidance.")


def test_csv_export(registry: SBMLParameterRegistry):
    output_path = ROOT_DIR / "results" / "roblitz_parameter_registry.csv"
    exported = registry.export_csv(output_path)
    assert exported.exists(), f"Export file was not created at {exported}"

    with open(exported, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 1 header + 141 parameters = 142 lines
    assert len(lines) == 142, f"Expected 142 lines in exported CSV, found {len(lines)}"
    print(f"[OK] Test 7: Parameter registry exported to '{exported.relative_to(ROOT_DIR)}' (142 lines).")


def main():
    print("=" * 65)
    print("PHASE 7.1: ROBLITZ SBML PARAMETER REGISTRY VERIFICATION")
    print("=" * 65)

    registry = get_sbml_parameter_registry()

    test_parameter_count(registry)
    test_unique_identifiers(registry)
    test_finite_values(registry)
    test_core_5_candidates(registry)
    test_bidirectional_lookup(registry)
    test_unknown_parameter_raises(registry)
    test_csv_export(registry)

    print("=" * 65)
    print("ALL PHASE 7.1 TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    main()
