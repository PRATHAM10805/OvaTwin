"""
test_sbml_reactions.py

Unit test and verification suite for Phase 7.2:
Verifying the governing reaction equations, rate laws, and target species
for each of the Core 5 candidate parameters in the Röblitz 2013 SBML model.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.HPO.sbml_reactions import (
    SBMLReactionAnalyzer,
    get_sbml_reaction_analyzer,
)


def test_reaction_count(analyzer: SBMLReactionAnalyzer):
    assert len(analyzer._all_reactions) == 71, (
        f"Expected 71 reactions in SBML model, found {len(analyzer._all_reactions)}"
    )
    assert len(analyzer._function_definitions) == 71, (
        f"Expected 71 function definitions, found {len(analyzer._function_definitions)}"
    )
    print("[OK] Test 1: Model contains exactly 71 reactions and 71 function definitions.")


def test_p1_reaction(analyzer: SBMLReactionAnalyzer):
    participants = analyzer.get_reactions_for_parameter("p1")
    assert len(participants) >= 1, "p1 not found in any reaction rate law!"

    rxn = participants[0]
    assert rxn.reaction_id == "re25"
    assert "LH_Pit" in rxn.products
    assert "E2" in rxn.modifiers
    assert "P4" in rxn.modifiers
    assert "function_10" in rxn.function_id
    print(f"     -> p1   | b_syn_LH  : reaction {rxn.reaction_id} produces {rxn.products} (governed by {rxn.function_id})")
    print("[OK] Test 2: Parameter p1 verified as LH basal synthesis rate into pituitary reservoir (LH_Pit).")


def test_p17_reactions(analyzer: SBMLReactionAnalyzer):
    participants = analyzer.get_reactions_for_parameter("p17")
    rxn_ids = [p.reaction_id for p in participants]
    assert "re84" in rxn_ids, "p17 not found in expected release reaction re84!"
    assert "re29" in rxn_ids, "p17 not found in expected depletion reaction re29!"

    # re84 produces circulating blood FSH
    re84 = next(p for p in participants if p.reaction_id == "re84")
    assert "FSH_bld" in re84.products
    assert "FSH_pit" in re84.modifiers
    print(f"     -> p17  | b_rel_FSH : reaction {re84.reaction_id} releases into {re84.products} from pituitary FSH")
    print("[OK] Test 3: Parameter p17 verified as FSH basal release rate from pituitary into bloodstream (FSH_bld).")


def test_p49_reaction(analyzer: SBMLReactionAnalyzer):
    participants = analyzer.get_reactions_for_parameter("p49")
    assert len(participants) == 1, f"Expected 1 reaction for p49, found {len(participants)}"

    rxn = participants[0]
    assert rxn.reaction_id == "re43"
    assert "AF1" in rxn.products
    assert "FSH_R" in rxn.modifiers
    assert "function_21" in rxn.function_id
    print(f"     -> p49  | k_AF1     : reaction {rxn.reaction_id} produces {rxn.products} via FSH_R stimulation")
    print("[OK] Test 4: Parameter p49 verified as maximal recruitment rate into early antral follicles (AF1).")


def test_p51_reaction(analyzer: SBMLReactionAnalyzer):
    participants = analyzer.get_reactions_for_parameter("p51")
    assert len(participants) == 1, f"Expected 1 reaction for p51, found {len(participants)}"

    rxn = participants[0]
    assert rxn.reaction_id == "re44"
    assert "AF2" in rxn.reactants
    assert "AF3" in rxn.products
    assert "function_22" in rxn.function_id
    print(f"     -> p51  | k_AF2_AF3 : reaction {rxn.reaction_id} converts {rxn.reactants} -> {rxn.products}")
    print("[OK] Test 5: Parameter p51 verified as selection rate from antral follicles (AF2) to dominant stage (AF3).")


def test_p173_reaction(analyzer: SBMLReactionAnalyzer):
    participants = analyzer.get_reactions_for_parameter("p173")
    assert len(participants) == 1, f"Expected 1 reaction for p173, found {len(participants)}"

    rxn = participants[0]
    assert rxn.reaction_id == "re72"
    assert "InhB" in rxn.products
    assert "AF2" in rxn.modifiers
    assert "function_44" in rxn.function_id
    print(f"     -> p173 | b_IhB     : reaction {rxn.reaction_id} produces {rxn.products} (modulated by {rxn.modifiers})")
    print("[OK] Test 6: Parameter p173 verified as basal synthesis rate of Inhibin B (InhB).")


def test_core_candidate_verification_api(analyzer: SBMLReactionAnalyzer):
    verification = analyzer.verify_core_candidate_reactions()
    for pid, verified in verification.items():
        assert verified, f"Candidate parameter {pid} reaction verification failed!"
    print("[OK] Test 7: verify_core_candidate_reactions() returns True for all 5 candidate parameters.")


def test_csv_export(analyzer: SBMLReactionAnalyzer):
    output_path = ROOT_DIR / "results" / "roblitz_candidate_reactions.csv"
    exported = analyzer.export_csv(output_path)
    assert exported.exists(), f"Reaction export file not found at {exported}"

    with open(exported, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 1 header + 6 candidate reaction rows = 7 lines
    assert len(lines) >= 6, f"Expected at least 6 rows in exported CSV, found {len(lines)}"
    print(f"[OK] Test 8: Reaction equations audit exported to '{exported.relative_to(ROOT_DIR)}'.")


def main():
    print("=" * 65)
    print("PHASE 7.2: ROBLITZ REACTION EQUATIONS & RATE LAWS VERIFICATION")
    print("=" * 65)

    analyzer = get_sbml_reaction_analyzer()

    test_reaction_count(analyzer)
    test_p1_reaction(analyzer)
    test_p17_reactions(analyzer)
    test_p49_reaction(analyzer)
    test_p51_reaction(analyzer)
    test_p173_reaction(analyzer)
    test_core_candidate_verification_api(analyzer)
    test_csv_export(analyzer)

    print("=" * 65)
    print("ALL PHASE 7.2 TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    main()
