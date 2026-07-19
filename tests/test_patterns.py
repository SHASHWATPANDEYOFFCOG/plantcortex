"""D6 tests: pattern cards, causal mining (N3), cause ranking, and eval smoke."""

from __future__ import annotations

from agents import m6_patterns as m6


def test_seal_pattern_card_with_vibration_precursor(ingested_repos):
    cards = m6.pattern_cards(ingested_repos.graph)
    seal = next(c for c in cards
                if c.equipment == "P-101A" and "seal" in c.failure_name)
    assert seal.count == 4                              # the 4 seeded seal failures
    assert seal.precursor.get("precursor_code") == "VIB"
    assert seal.precursor["matched"] >= 3              # vibration precedes them
    assert "vibration" in seal.recommendation.lower()
    # the strongest-precursor pattern should rank at/near the top
    assert cards.index(seal) == 0


def test_mine_causal_learns_has_cause_edges(ingested_repos):
    n = m6.mine_causal(ingested_repos.graph)
    assert n >= 1
    g = ingested_repos.graph
    has_cause = [(e["source"], e["target"]) for e in g.all_edges()
                 if e["type"] == "HAS_CAUSE"]
    # external leakage learned to be caused-by / preceded-by vibration
    assert ("FailureMode:ELP", "FailureMode:VIB") in has_cause


def test_cause_ranking_anticipates_vibration(ingested_repos):
    m6.mine_causal(ingested_repos.graph)
    r = m6.rank_causes(ingested_repos.graph,
                       "mechanical seal leak on P-101A, oil on baseplate")
    assert r["equipment"] == "P-101A" and r["observed_failure"] == "ELP"
    assert r["causes"] and r["causes"][0]["cause"] == "VIB"


def test_eval_extraction_and_compliance(ingested_repos):
    from eval.run import compliance_eval, extraction_recall
    ext = extraction_recall(ingested_repos)
    assert ext["overall"] > 60                          # core types fully covered
    node_types = {row[0]: row[3] for row in ext["nodes"]}
    assert node_types["Equipment"] == 100.0 and node_types["Procedure"] == 100.0
    comp = compliance_eval(ingested_repos)
    assert comp["seeded_caught"] and comp["false_positives"] == []
