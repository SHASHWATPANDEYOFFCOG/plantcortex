"""Tests for Technical Language Processing normalization."""

from __future__ import annotations

from pipelines.m1_ingest.tlp import normalize


def test_expands_common_shorthand():
    out, changes = normalize("chng seel pmp, brg noise hi vib")
    low = out.lower()
    assert "changed" in low and "seal" in low and "pump" in low
    assert "bearing" in low and "vibration" in low and "high" in low
    assert any(c["from"].lower() == "brg" for c in changes)


def test_preserves_equipment_tags():
    out, _ = normalize("mech seel lkg P-101A hi vib")
    assert "P-101A" in out                     # tag untouched
    assert "leakage" in out.lower()


def test_no_change_returns_input():
    out, changes = normalize("routine inspection completed")
    assert "inspection" in out.lower()
    assert isinstance(changes, list)


def test_empty():
    assert normalize("") == ("", [])
