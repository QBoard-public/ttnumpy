# Copyright (c) 2026 QBoard LLC. Licensed under the AGPLv3. See LICENSE.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run every script in the examples directory.

Each script asserts its results against a closed form or a reference
solution, so a script that runs to the end has passed its own checks.
"""

import contextlib
import io
import runpy
import unittest
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


class TestExamples(unittest.TestCase):
    """The scripts of the examples directory."""

    def test_every_example_runs_and_passes_its_checks(self):
        """Each script runs to the end, past the checks it asserts."""
        scripts = sorted(EXAMPLES.glob("*.py"))
        self.assertTrue(scripts, f"no example scripts in {EXAMPLES}")
        for script in scripts:
            with self.subTest(script=script.name):
                with contextlib.redirect_stdout(io.StringIO()):
                    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    unittest.main()
