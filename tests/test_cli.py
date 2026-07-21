"""CLI wiring: the parser exposes the ML commands and rejects unknown ones."""

import pytest

from futmarket import cli


def test_events_command_runs_on_empty_db(config, conn, tmp_path, capsys):
    # A read-only command that touches the DB but needs no network — proves the
    # parser dispatches and the config/db plumbing works end to end.
    conn.close()
    cli.main(["--config", str(tmp_path / "config.yaml"), "events"])
    assert "no events" in capsys.readouterr().out


def test_unknown_command_is_rejected():
    with pytest.raises(SystemExit):
        cli.main(["definitely-not-a-command"])


def test_removed_old_engine_commands_are_gone():
    # The old rules engine is gone; its commands must not resurface.
    for gone in ("backtest", "advise", "signals", "scan-momentum", "watchlist"):
        with pytest.raises(SystemExit):
            cli.main([gone])
