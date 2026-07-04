"""CLI behavior: manual logging, history output and its Δ% math."""

from datetime import datetime, timezone

from futmarket import cli
from futmarket import db as futdb


def test_log_price_and_history_delta(config, conn, tmp_path, capsys, monkeypatch):
    ts = [
        datetime(2026, 7, 1, 10, 0, tzinfo=timezone.utc),
        datetime(2026, 7, 1, 11, 0, tzinfo=timezone.utc),
        datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc),
    ]
    for at, price in zip(ts, [100_000, 110_000, 99_000]):
        assert futdb.insert_snapshot(conn, player_id="p1", price=price,
                                     source="manual", at=at)
    conn.commit()

    cli.main(["--config", str(tmp_path / "config.yaml"), "history", "p1"])
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 4  # header + 3 rows, oldest first
    assert "100,000" in out[1] and "+10.0%" in out[2] and "-10.0%" in out[3]


def test_log_price_rejects_unknown_player(config, tmp_path, capsys):
    import pytest
    with pytest.raises(SystemExit):
        cli.main(["--config", str(tmp_path / "config.yaml"), "log-price", "nobody", "5000"])


def test_config_rejects_oversized_watchlist(tmp_path):
    import pytest
    from futmarket.config import ConfigError, load_config
    entries = "\n".join(
        f'  - {{ player_id: p{i}, name: "P{i}", rating: 80, position: ST, version: "Base Gold" }}'
        for i in range(51)
    )
    cfg = tmp_path / "big.yaml"
    cfg.write_text(f"source: mock\nwatchlist:\n{entries}\n")
    with pytest.raises(ConfigError, match="cap"):
        load_config(cfg)
