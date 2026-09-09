"""Pytest configuration shared by all tests under this package."""

from __future__ import annotations

import pytest


def pytest_report_header(config: pytest.Config) -> str:
    """Extra line at the start of the pytest run (visible in CI logs)."""
    return f"benchmarks tests | rootdir: {config.rootpath}"
