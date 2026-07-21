"""The dashboard web layer: fragment renderers, read API, and guarded controls."""

import time

import pytest
from fastapi.testclient import TestClient

from futmarket.ml import advise, dashboard
from futmarket.webapp import create_app


# ---- fragment renderers (pure, synthetic) ---------------------------------

def _read(verdict="BUY", **kw):
    base = dict(verdict=verdict, headline="a plain read", price=12000,
                target=14000, stop=11000, reward_risk=1.5, expensive=False,
                on_dip=True, reasons=["on the dip", "room to bounce"])
    base.update(kw)
    return base


def test_render_card_reads_shows_verdict_and_trade():
    row = {"name": "Test Card", "rating": 84, "version": "Rare", "url": "u"}
    html = dashboard.render_card_reads([(row, _read("BUY"))])
    assert "BUY" in html and "Test Card" in html
    assert "14,000" in html and "11,000" in html          # sell + stop shown for BUY


def test_render_card_reads_avoid_has_no_trade_levels():
    row = {"name": "Knife", "rating": 90, "version": "TOTS", "url": None}
    html = dashboard.render_card_reads([(row, _read("AVOID"))])
    assert "AVOID" in html and 'class="levels"' not in html


def test_render_card_reads_empty():
    assert "No card matched" in dashboard.render_card_reads([])


def test_render_group_read():
    read = {"dim": "rating", "value": "84", "n": 12, "group_move_7d": 5.0,
            "share_on_dip": 25.0,
            "opportunities": [{"name": "A", "version": "Rare", "price": 800,
                               "confidence": 0.6, "url": None}]}
    html = dashboard.render_group_read(read)
    assert "rating 84" in html and "12 cards" in html and "A" in html


def test_render_scorecard_full_empty_and_filled():
    assert "none graded yet" in dashboard.render_scorecard_full(
        {"total": 3, "closed": 1, "graded": 0}, [])
    sc = {"graded": 2, "win_rate": 0.5, "return_on_capital_pct": 4.4,
          "coins_pnl": 1200, "avg_coins_per_trade": 600, "median_return_pct": 3.0}
    rows = [{"name": "X", "player_id": "x", "entry_price": 10000,
             "status": "target", "realized_pct": 12.0}]
    html = dashboard.render_scorecard_full(sc, rows)
    assert "+4.4%" in html and "X" in html and "target" in html


# ---- the web app ----------------------------------------------------------

@pytest.fixture
def client(tmp_path, config, monkeypatch):
    # keep the consult cache off the real data dir so the test stays hermetic
    monkeypatch.setattr(advise, "CACHE_PATH", tmp_path / "advise.pkl")
    return TestClient(create_app(str(tmp_path / "config.yaml")))


def test_app_js_served_with_js_type(client):
    r = client.get("/app.js")
    assert r.status_code == 200
    assert "javascript" in r.headers["content-type"]


def test_scorecard_and_picks_fragments(client):
    assert client.get("/api/scorecard").status_code == 200
    assert client.get("/api/picks").status_code == 200


def test_advise_on_empty_db_is_graceful(client):
    r = client.get("/api/advise", params={"q": "mbappe"})
    assert r.status_code == 200 and "No card matched" in r.json()["html"]


def test_cross_site_action_is_blocked(client):
    r = client.post("/api/run/scorecard", headers={"origin": "http://evil.com"})
    assert r.status_code == 403


def test_unknown_action_is_404(client):
    assert client.post("/api/run/nope").status_code == 404


def test_action_runs_to_completion(client):
    jid = client.post("/api/run/scorecard").json()["job_id"]
    for _ in range(40):
        j = client.get(f"/api/jobs/{jid}").json()
        if j["status"] in ("done", "error"):
            break
        time.sleep(0.5)
    assert j["status"] == "done"
