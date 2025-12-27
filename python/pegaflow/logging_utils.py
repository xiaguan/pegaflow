"""Logging utilities for PegaFlow connector.

This module provides timing decorators and logger configuration.
"""

import functools
import logging
import os
import time

# Environment variable to control timing logging
ENABLE_TIMING = os.environ.get("PEGAFLOW_ENABLE_TIMING", "1") == "1"

# Module logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    _handler = logging.StreamHandler()
    _handler.setLevel(logging.NOTSET)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.propagate = False


def get_connector_logger() -> logging.Logger:
    """Get a logger for the connector module."""
    connector_logger = logging.getLogger("pegaflow.connector")
    connector_logger.setLevel(logging.INFO)
    if not connector_logger.hasHandlers():
        handler = logging.StreamHandler()
        handler.setLevel(logging.NOTSET)
        handler.setFormatter(logging.Formatter("%(message)s"))
        connector_logger.addHandler(handler)
        connector_logger.propagate = False
    return connector_logger


def timing_wrapper(func):
    """Decorator to log function name and execution time when enabled.

    Enable by setting environment variable: PEGAFLOW_ENABLE_TIMING=1
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not ENABLE_TIMING:
            return func(*args, **kwargs)

        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug(
            "[PegaKVConnector] %s took %.2f ms",
            func.__name__,
            elapsed_ms,
        )
        return result

    return wrapper


__all__ = ["ENABLE_TIMING", "timing_wrapper", "get_connector_logger"]
