import logging
import os
from typing import Optional


_CONFIGURED = False


def _ensure_configured(default_level: int = logging.INFO) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level_name = os.getenv("CLIMATEGPT_LOG_LEVEL", "").upper()
    level: int = getattr(logging, level_name, default_level) if level_name else default_level
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    _CONFIGURED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    _ensure_configured()
    return logging.getLogger(name if name else "climategpt")












