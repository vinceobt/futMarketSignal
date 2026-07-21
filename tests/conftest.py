import textwrap

import pytest

from futmarket import db as futdb
from futmarket.config import load_config


@pytest.fixture
def config(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(textwrap.dedent("""
        source: futgg
        platform: console
        database_path: data/test.db
        log_path: data/test.log
        tax_rate: 0.05
        target_pct: 25.0
        stop_pct: 8.0
    """))
    return load_config(cfg)


@pytest.fixture
def conn(config):
    return futdb.connect(config.database_path)
