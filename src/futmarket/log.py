"""Structured logging: ISO timestamps to stderr, plus a rotating file in data/."""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
DATEFMT = "%Y-%m-%dT%H:%M:%S%z"


def setup_logging(log_path: str | Path, level: int = logging.INFO) -> logging.Logger:
    root = logging.getLogger("futmarket")
    if root.handlers:  # already configured (tests, repeated CLI calls)
        return root
    root.setLevel(level)

    stream = logging.StreamHandler()
    stream.setFormatter(logging.Formatter(FORMAT, DATEFMT))
    root.addHandler(stream)

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=2_000_000, backupCount=3
    )
    file_handler.setFormatter(logging.Formatter(FORMAT, DATEFMT))
    root.addHandler(file_handler)
    return root
