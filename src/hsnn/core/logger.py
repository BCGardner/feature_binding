import logging
from typing import Optional

_LOGGING_FORMAT = '[%(levelname)s] [%(name)s] %(message)s'

__all__ = ["get_logger"]


def get_logger(name: Optional[str] = None) -> logging.Logger:
    if logging.getLogger(name).hasHandlers():
        return logging.getLogger(name)

    formatter = logging.Formatter(_LOGGING_FORMAT)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.addHandler(handler)

    return logger
