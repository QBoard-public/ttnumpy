"""
Application-facing logging setup for the ``ttnumpy`` package.

The package uses exactly two loggers, both direct children of the root logger.
Reach them with the plain stdlib call and the names below:

* ``PACKAGE_LOGGER_NAME`` (``ttnumpy``) -- everything the library emits.
* ``TEST_LOGGER_NAME`` (``ttnumpy-test``) -- test code. The hyphen matters:
  logger hierarchy splits on dots, so ``ttnumpy-test`` is a separate root and
  raising the library's verbosity never raises the tests'.

Module identity is not lost by sharing one logger name. A filter installed at
the bottom of this module stamps every record with the dotted path of the file
that emitted it, so a line still reads ``[ttnumpy.solvers.common]``.

Library code attaches no handlers. Applications opt in by calling
:func:`setup_logging` once from a script or CLI; it configures the ``ttnumpy``
logger, never the root logger.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Union

# The one logger every ttnumpy module reports to.
PACKAGE_LOGGER_NAME = "ttnumpy"

# The one logger test code reports to. Not a child of PACKAGE_LOGGER_NAME:
# hierarchy splits on dots, so the hyphen keeps the two independent.
TEST_LOGGER_NAME = "ttnumpy-test"

# Environment variable consulted when setup_logging() is called without a level.
LOG_LEVEL_ENV_VAR = "TTNUMPY_LOG_LEVEL"

# Marks the handler installed by setup_logging(), so repeated calls replace it
# without disturbing handlers the host application attached itself.
_TTNUMPY_HANDLER_ATTR = "_ttnumpy_managed"

# Record attribute holding the emitting module's dotted path, relative to the
# package directory. Set by _ModulePathFilter.
MODULE_PATH_ATTR = "module_path"

# Directory of this file, i.e. the ttnumpy package root.
_PACKAGE_DIR = Path(__file__).resolve().parent

# Legacy verbosity scale, kept so existing setup_logging(0|1|2) calls still work.
_VERBOSITY_LEVELS: Dict[int, int] = {
    0: logging.WARNING,
    1: logging.INFO,
    2: logging.DEBUG,
}

# The logger name plus the emitting module's path, so a record still says which
# module spoke and stays easy to pick out of another library's output.
SIMPLE_FORMAT = "[%(name)s.%(module_path)s] %(levelname)s %(message)s"
DETAILED_FORMAT = (
    "[%(name)s.%(module_path)s] %(levelname)s %(asctime)s "
    "%(filename)s:%(lineno)d %(message)s"
)

LevelLike = Union[int, str, None]


@lru_cache(maxsize=None)
def _module_path(pathname: str) -> str:
    """Dotted path of a source file, relative to the ttnumpy package directory.

    ``/.../ttnumpy/solvers/common.py`` becomes
    ``solvers.common``. Files outside the package fall back to their
    stem. Cached because it runs for every record and the set of source files
    is small and fixed.
    """
    try:
        relative = Path(pathname).resolve().relative_to(_PACKAGE_DIR)
    except (ValueError, OSError):
        return Path(pathname).stem
    return ".".join(relative.with_suffix("").parts)


class _ModulePathFilter(logging.Filter):
    """Stamps each record with the dotted path of the module that emitted it.

    Attached to the two package loggers at import time. It is a filter rather
    than a formatter so the attribute is present before the record
    propagates, which keeps it available to handlers this package does not
    own -- pytest's capture handlers in particular.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        setattr(record, MODULE_PATH_ATTR, _module_path(record.pathname))
        return True


class LevelFormatter(logging.Formatter):
    """
    A logging formatter that selects a different formatter per logging level.
    """

    def __init__(
        self,
        formatters: Dict[int, logging.Formatter],
        default_formatter: Optional[logging.Formatter] = None,
    ):
        """
        Initialize the LevelFormatter with specified formatters and a default formatter.

        Args:
            formatters (dict): A dictionary mapping logging levels to formatters.
            default_formatter (logging.Formatter, optional): The default formatter to use if no specific formatter is found.
        """
        super().__init__()
        self.formatters = formatters
        self.default_formatter = default_formatter

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the specified record using the appropriate formatter based on the record's level.

        Args:
            record (logging.LogRecord): The log record to format.

        Returns:
            str: The formatted log message.
        """
        formatter = self.formatters.get(record.levelno, self.default_formatter)
        if formatter is None:
            raise RuntimeError(
                f"No formatter for level {record.levelno} and no default_formatter"
            )
        return formatter.format(record)


def _level_names_mapping() -> Dict[str, int]:
    """
    Return the level-name to level-number mapping.

    ``logging.getLevelNamesMapping`` is the supported reverse lookup, but it
    only exists on Python 3.11+; the explicit table covers the 3.10 floor this
    package still declares. ``logging.getLevelName`` is deliberately not used:
    its ``str -> int`` direction is deprecated.

    Returns:
        Dict[str, int]: Level names mapped to their numeric levels.
    """
    getter = getattr(logging, "getLevelNamesMapping", None)
    if getter is not None:
        return getter()
    # FATAL and WARN are aliases of CRITICAL and WARNING; spelling them out
    # here avoids referencing the deprecated logging.FATAL / logging.WARN.
    return {
        "CRITICAL": logging.CRITICAL,
        "FATAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "WARN": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "NOTSET": logging.NOTSET,
    }


def resolve_level(level: LevelLike = None) -> int:
    """
    Translate the accepted level spellings into a ``logging`` level number.

    Accepts a level name (``"DEBUG"``), a ``logging`` constant
    (``logging.DEBUG``), or a legacy verbosity count (``0`` WARNING, ``1`` INFO,
    ``2`` or more DEBUG). ``None`` reads ``TTNUMPY_LOG_LEVEL`` from the environment
    and falls back to ``WARNING``.

    Args:
        level: The level to resolve.

    Returns:
        int: The corresponding ``logging`` level number.

    Raises:
        ValueError: If the level is not a recognised name or number.
    """
    if level is None:
        env_level = os.environ.get(LOG_LEVEL_ENV_VAR)
        if env_level is None or not env_level.strip():
            return logging.WARNING
        return resolve_level(env_level.strip())

    if isinstance(level, bool):
        raise ValueError(f"Invalid log level: {level!r}")

    if isinstance(level, int):
        # Small integers are the legacy verbosity scale; anything at or above
        # logging.DEBUG (10) is already a level number.
        if level in _VERBOSITY_LEVELS:
            return _VERBOSITY_LEVELS[level]
        if level < logging.DEBUG:
            return logging.DEBUG
        return level

    name = level.strip().upper()
    if name.isdigit():
        return resolve_level(int(name))

    resolved = _level_names_mapping().get(name)
    if resolved is None:
        raise ValueError(f"Unknown log level name: {level!r}")
    return resolved


def set_level(level: LevelLike = None) -> int:
    """
    Set the verbosity of the ``ttnumpy`` logger without touching its handlers.

    Args:
        level: Any level accepted by :func:`resolve_level`.

    Returns:
        int: The ``logging`` level number that was applied.
    """
    resolved = resolve_level(level)
    logging.getLogger(PACKAGE_LOGGER_NAME).setLevel(resolved)
    return resolved


def setup_logging(
    level: LevelLike = None,
    *,
    stream=None,
    capture_warnings: bool = False,
) -> logging.Logger:
    """
    Send ``ttnumpy`` log records to a stream handler.

    Safe to call repeatedly: the handler installed by a previous call is
    replaced rather than stacked, and handlers attached by the host application
    are left alone. The root logger is never modified, so third-party libraries
    keep whatever verbosity they already had.

    Args:
        level: Any level accepted by :func:`resolve_level`. ``None`` reads
            ``TTNUMPY_LOG_LEVEL`` and falls back to ``WARNING``.
        stream: Destination stream. Defaults to ``sys.stderr``.
        capture_warnings (bool): Route :mod:`warnings` through logging. This is
            a process-wide setting, so it stays off unless asked for. Enable it
            only when the application has configured the root logger: warnings
            are logged to ``py.warnings``, which sits outside the ``ttnumpy`` tree,
            and :func:`logging.captureWarnings` gives it a ``NullHandler``, so
            without a root handler the warnings are discarded rather than shown.

    Returns:
        logging.Logger: The configured ``ttnumpy`` package logger.
    """
    log_level = set_level(level)
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)

    for existing in [
        h for h in logger.handlers if getattr(h, _TTNUMPY_HANDLER_ATTR, False)
    ]:
        logger.removeHandler(existing)
        existing.close()

    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.DEBUG)
    setattr(handler, _TTNUMPY_HANDLER_ATTR, True)
    # SIMPLE_FORMAT and DETAILED_FORMAT reference module_path, so stamp it here
    # too: the logger's filter covers records emitted through it, and this
    # covers the handler being reused for records that arrived another way.
    handler.addFilter(_ModulePathFilter())

    if log_level <= logging.DEBUG:
        simple_formatter = logging.Formatter(SIMPLE_FORMAT)
        detailed_formatter = logging.Formatter(DETAILED_FORMAT)
        handler.setFormatter(
            LevelFormatter(
                {
                    logging.DEBUG: simple_formatter,
                    logging.INFO: simple_formatter,
                    logging.WARNING: detailed_formatter,
                    logging.ERROR: detailed_formatter,
                    logging.CRITICAL: detailed_formatter,
                },
                default_formatter=simple_formatter,
            )
        )
    else:
        handler.setFormatter(logging.Formatter(SIMPLE_FORMAT))

    logger.addHandler(handler)
    # We own the output now; without this a host application's root handler
    # would print every ttnumpy record a second time.
    logger.propagate = False

    if capture_warnings:
        logging.captureWarnings(True)

    return logger


def reset_logging() -> logging.Logger:
    """
    Undo :func:`setup_logging`, restoring the quiet library default.

    Removes the handler installed by :func:`setup_logging`, restores
    propagation, and clears the explicit level.

    Returns:
        logging.Logger: The ``ttnumpy`` package logger.
    """
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    for existing in [
        h for h in logger.handlers if getattr(h, _TTNUMPY_HANDLER_ATTR, False)
    ]:
        logger.removeHandler(existing)
        existing.close()
    logger.propagate = True
    logger.setLevel(logging.NOTSET)
    return logger


# Installed at import so every record from the two package loggers carries
# module_path. Modules reach the loggers with logging.getLogger(...); importing
# either name from this module is what guarantees this ran first.
for _logger_name in (PACKAGE_LOGGER_NAME, TEST_LOGGER_NAME):
    logging.getLogger(_logger_name).addFilter(_ModulePathFilter())
