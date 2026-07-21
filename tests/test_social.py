"""Social buzz: name matching, sentiment lean, and storage.

Matching is the risky part. A wrong match teaches the model that chatter moved a
card nobody mentioned, which is worse than learning nothing, so these tests lean
on what must NOT match.
"""

from datetime import datetime, timezone

import pytest

from futmarket import db as futdb, secrets
from futmarket.collectors import social_sources
from futmarket.services import social

UTC = timezone.utc
NOW = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)


def _card(conn, pid, name, **kw):
    futdb.upsert_card_meta(conn, {"player_id": pid, "name": name,
                                  "tradeable": kw.get("tradeable", 1),
                                  "rating": kw.get("rating", 85)})


# ---- name matching --------------------------------------------------------

def test_matches_a_surname(conn):
    _card(conn, "c1", "Erling Haaland")
    conn.commit()
    idx = social.build_name_index(conn)
    assert social.find_mentions("Haaland is going to moon this week", idx) == {"c1"}


def test_accents_do_not_block_a_match(conn):
    _card(conn, "c1", "João Félix")
    conn.commit()
    idx = social.build_name_index(conn)
    assert social.find_mentions("felix looks cheap right now", idx) == {"c1"}


def test_hype_attaches_to_every_card_of_that_player(conn):
    """Talk about a player lifts all his cards, not one arbitrary version."""
    _card(conn, "felix-tots", "João Félix")
    _card(conn, "felix-totw", "João Félix")
    conn.commit()
    idx = social.build_name_index(conn)
    assert social.find_mentions("felix is rising", idx) == {"felix-tots", "felix-totw"}


def test_ambiguous_surnames_are_dropped(conn):
    """'Rodriguez' names four different players, so it identifies nobody."""
    for i, first in enumerate(("James", "Guido", "Jose", "Luis")):
        _card(conn, f"r{i}", f"{first} Rodriguez")
    conn.commit()
    idx = social.build_name_index(conn)
    assert social.find_mentions("Rodriguez is cheap", idx) == set()


def test_short_names_are_ignored(conn):
    _card(conn, "c1", "Heung Min Son")
    conn.commit()
    idx = social.build_name_index(conn)
    # "son" would fire on "my son", "Sonic", every other sentence
    assert social.find_mentions("my son loves this game", idx) == set()


def test_football_vocabulary_is_not_a_player(conn):
    _card(conn, "c1", "Team Chemistry")     # contrived, but the words are common
    conn.commit()
    idx = social.build_name_index(conn)
    assert social.find_mentions("my squad chemistry is bad", idx) == set()


def test_substrings_do_not_count(conn):
    _card(conn, "c1", "Antony Martial")
    conn.commit()
    idx = social.build_name_index(conn)
    assert social.find_mentions("martials", idx) == set()      # word boundary
    assert social.find_mentions("Martial is up", idx) == {"c1"}


def test_untradeable_cards_are_not_indexed(conn):
    _card(conn, "sbc", "Erling Haaland", tradeable=0)
    conn.commit()
    assert social.find_mentions("haaland", social.build_name_index(conn)) == set()


# ---- sentiment lean -------------------------------------------------------

def test_lean_reads_trader_vocabulary():
    assert social_sources.lean("BUY these, huge profit, undervalued") > 0.5
    assert social_sources.lean("dump it now, this is crashing hard") < -0.5
    assert social_sources.lean("thoughts on this squad?") == 0.0


def test_lean_balances_mixed_talk():
    assert social_sources.lean("some say buy, others say sell") == 0.0


# ---- collection + storage -------------------------------------------------

def test_collect_stores_signal_per_card(conn):
    _card(conn, "c1", "Erling Haaland")
    _card(conn, "c2", "Kylian Mbappe")
    conn.commit()
    posts = [
        social_sources.Post("reddit", "Haaland is a great invest, buy now", NOW, 40),
        social_sources.Post("reddit", "Haaland rising again", NOW, 10),
        social_sources.Post("reddit", "nothing about anyone here", NOW, 3),
    ]
    res = social.collect(conn, posts=posts, now=NOW)
    assert res["posts"] == 3 and res["matched"] == 2 and res["cards"] == 1
    rows = futdb.sentiment_for(conn, "c1")
    assert rows[0]["mention_count"] == 2 and rows[0]["sentiment"] > 0


def test_collect_with_no_matches_writes_nothing(conn):
    _card(conn, "c1", "Erling Haaland")
    conn.commit()
    res = social.collect(conn, posts=[
        social_sources.Post("reddit", "best formation for FC 26?", NOW)], now=NOW)
    assert res["matched"] == 0 and res["cards"] == 0


# ---- x session from pasted cookies ----------------------------------------

def test_cookie_session_is_a_valid_storage_state(tmp_path):
    from futmarket.collectors import x_source
    path = x_source.build_session_from_cookies(
        "AUTH123", "CT0456", session_file=tmp_path / "x.json")
    import json
    state = json.loads(path.read_text())
    names = {(c["name"], c["domain"]) for c in state["cookies"]}
    assert ("auth_token", ".x.com") in names       # the session itself
    assert ("ct0", ".x.com") in names              # the CSRF token X pairs with it
    assert state["origins"] == []                  # playwright storage_state shape


def test_cookie_session_requires_auth_token(tmp_path):
    from futmarket.collectors import x_source
    from futmarket.collectors.base import SourceError
    with pytest.raises(SourceError) as e:
        x_source.build_session_from_cookies("  ", session_file=tmp_path / "x.json")
    assert "auth_token" in str(e.value)


# ---- credentials ----------------------------------------------------------

def test_missing_credentials_explain_where_to_get_them(monkeypatch):
    monkeypatch.setattr(secrets, "_loaded", True)
    monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
    monkeypatch.delenv("REDDIT_CLIENT_SECRET", raising=False)
    with pytest.raises(secrets.MissingCredentials) as e:
        secrets.require("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET")
    msg = str(e.value)
    assert "reddit.com/prefs/apps" in msg and "free" in msg.lower()


def test_env_file_does_not_override_real_environment(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('REDDIT_CLIENT_ID=from_file\nYOUTUBE_API_KEY="quoted"\n# comment\n')
    monkeypatch.setenv("REDDIT_CLIENT_ID", "from_environment")
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    secrets.load_env(env)
    import os
    assert os.environ["REDDIT_CLIENT_ID"] == "from_environment"   # deployment wins
    assert os.environ["YOUTUBE_API_KEY"] == "quoted"              # quotes stripped
