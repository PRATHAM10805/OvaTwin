"""
test_parameters_classification.py

Unit test and verification suite for Phase 7.3:
Verifying parameter categorization, metadata, hard safety bounds, and clamping.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.HPO.parameters import (
    AuthoritativeParameterRegistry,
    ParameterCategory,
    ParameterMetadata,
    get_authoritative_registry,
    HPOParameters,
)


def test_registry_size(registry: AuthoritativeParameterRegistry):
    assert len(registry) == 141, f"Expected 141 parameters, found {len(registry)}"
    print("[OK] Test 1: Authoritative registry contains exactly 141 parameters.")


def test_category_breakdown(registry: AuthoritativeParameterRegistry):
    core = registry.get_by_category(ParameterCategory.PERSONALIZABLE_CORE)
    exploratory = registry.get_by_category(ParameterCategory.EXPLORATORY)
    drugs = registry.get_by_category(ParameterCategory.DRUG_PK)
    structural = registry.get_by_category(ParameterCategory.STRUCTURAL)

    assert len(core) == 5, f"Expected 5 Core parameters, found {len(core)}"
    assert len(exploratory) == 6, f"Expected 6 Exploratory parameters, found {len(exploratory)}"
    assert len(drugs) == 21, f"Expected 21 Drug PK parameters, found {len(drugs)}"
    assert len(structural) == 109, f"Expected 109 Structural parameters, found {len(structural)}"
    assert len(core) + len(exploratory) + len(drugs) + len(structural) == 141

    print(f"     -> Core Personalizable: {len(core)}")
    print(f"     -> Exploratory         : {len(exploratory)}")
    print(f"     -> Drug PK             : {len(drugs)}")
    print(f"     -> Structural          : {len(structural)}")
    print("[OK] Test 2: Category distribution verified across all 141 parameters (5 + 6 + 21 + 109 = 141).")


def test_core_5_metadata(registry: AuthoritativeParameterRegistry):
    core_dict = registry.get_core_personalizable()
    expected_ids = {"p1", "p17", "p49", "p51", "p173"}
    assert set(core_dict.keys()) == expected_ids, f"Core IDs mismatch: {set(core_dict.keys())}"

    for pid, meta in core_dict.items():
        assert meta.category == ParameterCategory.PERSONALIZABLE_CORE
        assert meta.is_personalizable is True
        assert meta.min_bound < meta.baseline_value < meta.max_bound
        # check ratios (0.40 and 2.50)
        assert abs(meta.min_bound - 0.40 * meta.baseline_value) < 1e-6
        assert abs(meta.max_bound - 2.50 * meta.baseline_value) < 1e-6
        print(f"     -> {meta.id:5s} | {meta.name:12s} | Base: {meta.baseline_value:<12.4f} | Bounds: [{meta.min_bound:.4f}, {meta.max_bound:.4f}]")

    print("[OK] Test 3: Tier-1 Core 5 candidate parameters verified with exact safety bounds.")


def test_clamping_logic(registry: AuthoritativeParameterRegistry):
    p1_meta = registry.get("p1")
    base = p1_meta.baseline_value

    # Value within bounds should remain unchanged
    normal_val = base * 1.25
    assert p1_meta.clamp(normal_val) == normal_val
    assert registry.clamp("p1", normal_val) == normal_val

    # Value above max_bound should clamp to max_bound
    huge_val = base * 10.0
    assert p1_meta.clamp(huge_val) == p1_meta.max_bound
    assert registry.clamp("p1", huge_val) == p1_meta.max_bound

    # Value below min_bound should clamp to min_bound
    tiny_val = base * 0.10
    assert p1_meta.clamp(tiny_val) == p1_meta.min_bound
    assert registry.clamp("p1", tiny_val) == p1_meta.min_bound

    # Negative value should clamp to min_bound (positive safety)
    assert p1_meta.clamp(-500.0) == p1_meta.min_bound

    # NaN / Inf should fall back safely to baseline
    import math
    assert p1_meta.clamp(float("nan")) == base
    assert p1_meta.clamp(float("inf")) == base

    print("[OK] Test 4: Hard safety clamping logic verified for in-bounds, above, below, negative, and NaN/Inf.")


def test_bidirectional_lookup(registry: AuthoritativeParameterRegistry):
    assert registry.get("p1") == registry.get("b_syn_LH")
    assert registry.get("p17") == registry.get("b_rel_FSH")
    assert registry.get("p49") == registry.get("k_AF1")
    assert registry.get("p51") == registry.get("k_AF2_AF3")
    assert registry.get("p173") == registry.get("b_IhB")

    # Bounds lookup
    min_b, max_b = registry.get_bounds("b_syn_LH")
    assert min_b == registry.get("p1").min_bound
    assert max_b == registry.get("p1").max_bound

    print("[OK] Test 5: Bidirectional lookup and bounds retrieval verified across IDs and biological names.")


def test_csv_export(registry: AuthoritativeParameterRegistry):
    output_path = ROOT_DIR / "results" / "roblitz_parameter_classification.csv"
    exported = registry.export_csv(output_path)
    assert exported.exists(), f"Classification export file not found at {exported}"

    with open(exported, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 1 header + 141 parameters = 142 lines
    assert len(lines) == 142, f"Expected 142 lines in classification CSV, found {len(lines)}"
    print(f"[OK] Test 6: Parameter classification exported to '{exported.relative_to(ROOT_DIR)}' (142 lines).")


def test_legacy_dataclass():
    hpo = HPOParameters(fsh_production=5.5, lh_production=6.2)
    assert hpo.fsh_production == 5.5
    assert hpo.lh_production == 6.2
    assert hpo.gnrh_production is None
    print("[OK] Test 7: Legacy HPOParameters dataclass preserved and functional.")


def main():
    print("=" * 65)
    print("PHASE 7.3: ROBLITZ PARAMETER CLASSIFICATION & BOUNDS VERIFICATION")
    print("=" * 65)

    registry = get_authoritative_registry()

    test_registry_size(registry)
    test_category_breakdown(registry)
    test_core_5_metadata(registry)
    test_clamping_logic(registry)
    test_bidirectional_lookup(registry)
    test_csv_export(registry)
    test_legacy_dataclass()

    print("=" * 65)
    print("ALL PHASE 7.3 TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    main()
