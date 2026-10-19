# Contributing

Thank you for your time and interest in contributing to this project.
TTNumPy is a tensor-train toolbox for numerical linear algebra: a `TensorTrain`
container for the TT and TTM formats, constructors and quantization, exact
arithmetic with controlled rounding, and the ALS, MALS and AMEn solvers for
linear systems in tensor-train format.

We welcome all contributions including:

- bug fixes;
- improvements and features;
- documentation updates;
- test coverage improvements.

Every contribution helps improve the project and is greatly appreciated.

Before contributing to this project you should see [README](README.md).

---

## Contents

- [Development Setup](#development-setup)
- [Code Style and Formatting](#code-style-and-formatting)
- [Tests](#tests)
- [Documentation](#documentation)
- [Commit Messages](#commit-messages)
- [Creating a Merge Request](#creating-a-merge-request)
- [Continuous Integration (CI) Requirements](#continuous-integration-ci-requirements)
- [Review Process](#review-process)
- [Need Help?](#need-help)

---

## Development Setup

TTNumPy requires Python 3.11 or newer. Install the editable package with the
`dev` dependency group, which brings in pytest, Black, pylint and the
documentation toolchain. Dependency groups need pip 25.1 or newer:

```bash
python -m pip install -U pip
python -m pip install -e . --group dev
```

Verify the installation:

```bash
python -c "import ttnumpy as tt; print(tt.eye([2, 2]))"
```

---

## Code Style and Formatting

All contributions should:

- follow the project's coding conventions;
- produce clean, readable, and maintainable code;
- include appropriate comments where necessary;
- preserve backward compatibility whenever possible.

Please format your code and run all available validation tools before
submitting changes.

**Python guidelines**

Follow PEP 8 and format code with Black. The project's line length and target
version are configured in `pyproject.toml`, so running Black with no arguments
applies them:

```bash
black .
pylint $(git ls-files '*.py') --max-line-length 160 --fail-under 9 -d invalid-name
```

Public functions, classes and methods carry Google-style docstrings, which
Sphinx renders into the API reference. When adding or changing one, keep the
continuation lines of `Args:` and `Raises:` entries indented under their field
name — an unindented continuation is parsed as a new field and silently
corrupts the rendered documentation.

**Numerical code**

This library is about controlled approximation, so accuracy claims are part of
the interface:

- a parameter named as an error bound must be an upper bound on the achieved
  error, not a heuristic threshold;
- state which norm a tolerance refers to, and whether it is absolute or
  relative;
- when an operation changes tensor-train ranks, say so in the docstring, since
  cost grows with the cube of the rank.

**Branch Naming**

Create a dedicated branch for every contribution.
Recommended naming convention:

```
feature/<short_description>
bugfix/<short_description>
docs/<short_description>
hotfix/<short_description>
```

Do not commit directly to the `main` branch.

---

## Tests

The suite runs with `pytest` from the repository root:

```bash
pytest
```

New behaviour should arrive with tests, and a bug fix should come with a test
that fails without it. Tests live in `tests/` at the repository root, outside
the package, so they are not installed with it. The benchmark runner keeps its
own tests in `benchmarks/test`.

Seed anything random. The suite compares against exact expectations, so an
unseeded draw turns a real failure into an unreproducible one.

---

## Documentation

The `docs/` directory holds the Sphinx sources. Build them with:

```bash
python -m sphinx -b html -W docs docs/_build/html
```

`-W` turns warnings into errors, and `conf.py` sets `nitpicky = True`, so an
unresolvable cross-reference fails the build. Run it before opening a merge
request when your change touches docstrings or documentation.

Code examples in the documentation are expected to run as written. If you add
one, execute it first.

---

## Commit Messages

Write clear and descriptive commit messages.
Recommended examples:

```
feature: add rank-adaptive stopping to the AMEn solver

fix: bound the discarded tail in SVD truncation

docs: expand the quantization chapter

test: cover the truncation error bound

ci: update CI configuration
```

---

## Creating a Merge Request

When your work is ready:

1. Push your branch to the project repository.
2. Open a merge request targeting the `main` branch.
3. Provide a clear description of your changes.

Your merge request should include:

- a summary of the changes;
- testing performed;
- related issues, if any.

---

## Continuous Integration (CI) Requirements

Every merge request is automatically validated by the project's Continuous
Integration (CI) pipeline.

Before requesting a review, ensure that:

- the package imports and the documentation builds without warnings;
- all automated tests pass;
- `black --check` reports no changes;
- `pylint` stays at or above the configured score;
- documentation is updated when applicable.

Merge requests with failing CI checks will not be merged until all required
checks have passed.

---

## Review Process

All code contributions are reviewed by the project maintainers before being
merged.

During the review process, maintainers evaluate:

- correctness of the implementation, including the accuracy claims it makes;
- compliance with the project architecture;
- code quality and readability;
- test coverage;
- documentation updates.

Contributors are expected to address review comments and update their merge
requests accordingly.

A merge request may be merged only after receiving the required approvals and
successfully passing all mandatory CI checks.

---

## Licensing of Contributions

TTNumPy is distributed under the GNU Affero General Public License v3.0
(see [LICENSE](LICENSE)). By submitting a contribution you agree that it is
licensed under the same terms.

---

## Need Help?

If you have questions about contributing, development, or the project
workflow, please contact us at this email address: info@qboard.tech.
