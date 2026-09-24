# Copyright (c) 2026 QBoard LLC. Licensed under the AGPLv3. See LICENSE.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pytest configuration shared by all tests under this package."""

from __future__ import annotations

import pytest


def pytest_report_header(config: pytest.Config) -> str:
    """Extra line at the start of the pytest run (visible in CI logs)."""
    return f"benchmarks tests | rootdir: {config.rootpath}"
