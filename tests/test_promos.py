"""Promo classification: turn a calendar note into a normalised type."""

from futmarket.ml import promos


def test_classifies_the_big_promos():
    assert promos.classify("Champion ICON (15 cards)") == "icon"
    assert promos.classify("Heroes (93 cards)") == "hero"
    assert promos.classify("Team of the Season So Far") == "tots"
    assert promos.classify("Team of the Year") == "toty"


def test_classifies_sbc_and_totw():
    assert promos.classify("SBC: Enzo Fernández (Players)") == "sbc_player"
    assert promos.classify("SBC: 85+ Upgrade (Upgrades)") == "sbc_upgrade"
    assert promos.classify("Team of the Week (2025-09-17)") == "totw"


def test_falls_back_to_event_type_then_other():
    assert promos.classify("", event_type="PATCH") == "patch"
    assert promos.classify(None) == "other"
    assert promos.classify("something unrecognised") == "other"


def test_major_types():
    assert promos.is_major("Champion ICON")
    assert promos.is_major("Heroes")
    assert not promos.is_major("SBC: Enzo (Players)")
    assert not promos.is_major("Team of the Week")
