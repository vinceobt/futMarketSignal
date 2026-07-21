"""Alert helpers: the card-identity tag (so the wrong version can't be bought)
and the coin formatter used in the Discord run summary."""

from futmarket.alerts import _card_tag, _coins


def test_card_tag_combines_rating_and_version():
    assert _card_tag(94, "Team of the Season") == " · 94 Team of the Season"


def test_card_tag_partial_and_empty():
    assert _card_tag(94, None) == " · 94"
    assert _card_tag(None, "Winter Wildcards") == " · Winter Wildcards"
    assert _card_tag(None, None) == ""


def test_coins_formats_thousands_and_millions():
    assert _coins(162_000) == "162K"
    assert _coins(1_200_000) == "1.20M"
    assert _coins(500) == "500"
