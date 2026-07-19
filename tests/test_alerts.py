"""Trade-alert formatting: the BUY/SELL line names the exact card (rating +
version/rarity) so the wrong version can't be bought, and degrades cleanly when
that metadata is unknown."""

from dataclasses import dataclass

from futmarket.alerts import _card_tag, format_trade_alert


@dataclass
class _View:
    floor: float = 160_000.0
    bounces: int = 4


def test_card_tag_combines_rating_and_version():
    assert _card_tag(94, "Team of the Season") == " · 94 Team of the Season"


def test_card_tag_partial_and_empty():
    assert _card_tag(94, None) == " · 94"
    assert _card_tag(None, "Winter Wildcards") == " · Winter Wildcards"
    assert _card_tag(None, None) == ""


def test_buy_alert_includes_rating_and_version():
    msg = format_trade_alert("BUY", "Federico Valverde", 162_000, view=_View(),
                             target=213_000, rating=94, version="Team of the Season")
    assert "Federico Valverde · 94 Team of the Season" in msg
    assert "@ 162K" in msg and "target 213K" in msg


def test_sell_alert_includes_rating_and_version():
    msg = format_trade_alert("SELL", "Daniel Munoz", 1_200_000, realized_pct=8.3,
                             reason="hit +25% target", rating=96, version="RTTF")
    assert "Daniel Munoz · 96 RTTF" in msg
    assert "+8.3% net" in msg


def test_alert_without_metadata_is_unchanged():
    msg = format_trade_alert("BUY", "Unknown Card", 50_000, view=_View(), target=65_000)
    assert msg.startswith("🟢 BUY Unknown Card @ 50K")
    assert " · " not in msg
