"""Pytest configuration shared by every test package.

Holds the test-side logging setup. Test code logs through the ``ttnumpy-test``
logger, kept separate from the library's ``ttnumpy`` logger so a log line says
whether the library or the harness produced it, and so raising the library's
verbosity never raises the tests'.
"""

from __future__ import annotations

import logging

import pytest

from ttnumpy.logger_config import TEST_LOGGER_NAME

logger = logging.getLogger(TEST_LOGGER_NAME)


@pytest.fixture(autouse=True)
def log_test_boundaries(request: pytest.FixtureRequest):
    """Bracket every test with a ``ttnumpy-test`` record.

    At DEBUG this interleaves with the library's own output, so it is clear
    which test produced which records:

        [ttnumpy-test] conftest.py:31 DEBUG start ttnumpy/test/test_mals.py::test_tt_mals
        [ttnumpy]   tensor_train.py:153 DEBUG Tensor Train with rank 5 saved.
        [ttnumpy-test] conftest.py:33 DEBUG end   ttnumpy/test/test_mals.py::test_tt_mals

    Quiet on a normal run: these are DEBUG records, and without
    ``--log-cli-level`` nothing below WARNING is shown.
    """
    logger.debug("start %s", request.node.nodeid)
    yield
    logger.debug("end   %s", request.node.nodeid)
