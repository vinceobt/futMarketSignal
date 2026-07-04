import textwrap

import pytest

from futmarket import db as futdb
from futmarket.config import load_config


@pytest.fixture
def config(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(textwrap.dedent("""
        source: mock
        platform: console
        poll_minutes: 45
        inter_player_delay_seconds: [0, 0]
        skip_guard_fraction: 0.5
        max_consecutive_failures: 3
        cooldown_minutes: 60
        database_path: data/test.db
        log_path: data/test.log
        watchlist:
          - { player_id: p1, name: "Player One", rating: 88, position: ST, version: "Base Gold" }
          - { player_id: p2, name: "Player Two", rating: 90, position: CM, version: "Base Gold" }
          - { player_id: p3, name: "Player Three", rating: 85, position: CB, version: "Base Gold" }
    """))
    return load_config(cfg)


@pytest.fixture
def conn(config):
    return futdb.connect(config.database_path)
