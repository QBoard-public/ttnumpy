# TTNumPy

**Tensor-train toolbox for numerical linear algebra.**

TTNumPy provides tools for solving optimization problems in the tensor-train
formalism. Problems that this formalism handles efficiently include systems of
linear equations, black-box function interpolation and numerical integration.

The library works with tensor trains (TT) and tensor train matrices (TTM),
known in the physics literature as matrix product states and matrix product
operators. It is written in Python on top of NumPy and SciPy and pulls in no
other runtime dependencies.

## Features

- **Tensor train container.** `TensorTrain` holds both the TT and the TTM
  format, tracking bond dimensions, canonical form and the position of the
  orthogonality centre.
- **Constructors.** TT-SVD decomposition of a full tensor, plus `ones`,
  `zeros`, `eye`, `rand` and `canonical_tt`.
- **Arithmetic.** Addition, element-wise and scalar multiplication,
  matrix-by-vector and matrix-by-matrix products, transposition and
  conjugation, each with optional rounding.
- **Quantization.** Conversion between TT and quantized tensor train (QTT)
  form, which turns a grid of $2^n$ points into $n$ cores of mode size 2.
- **Linear solvers.** ALS, MALS and AMEn for $Ax = b$ in TT format, sharing
  environments, local solvers and stopping criteria.

## Installation

TTNumPy requires Python 3.10 or newer. Its only runtime dependencies are
NumPy and SciPy .

```bash
python -m pip install .
```

For development work, install the editable package with the developer extras:

```bash
python -m pip install -e '.[dev]'
```

The `docs` extra installs only the documentation toolchain:

```bash
python -m pip install -e '.[docs]'
```

The `benchmark` extra installs all needed packages for benchmarking:

```bash
python -m pip install -e '.[benchmark]'
```
This adds PyAMG and PETSc on top of the runtime requirements. PETSc has no
prebuilt wheels and is compiled from source, so it needs BLAS and LAPACK
development headers on the system — on Debian or Ubuntu:

```bash
sudo apt-get install -y libopenblas-dev liblapack-dev
```

Check the installation:

```bash
python -c "import ttnumpy as tt; print(tt.eye([2, 2]))"
```

## Quick start

Build a tensor train, combine trains, and solve a linear system in TT format.

```python
import numpy as np
import ttnumpy as tt

# decompose a full tensor
tensor = np.arange(24, dtype=float).reshape(2, 3, 4)
train = tt.tt_svd(tensor, error=1e-10)
print(train.bonds)                       # [1, 2, 2, 1]

# arithmetic stays in the format
left = tt.rand([2, 3, 4], ranks=3, seed=1)
right = tt.rand([2, 3, 4], ranks=3, seed=2)
combined = left + right                  # ranks add, then round
scaled = 0.5 * left

# solve A x = b with the rank-adaptive AMEn solver
A = tt.eye([2, 2, 2])                    # TTM operator
b = tt.rand([2, 2, 2], ranks=2, seed=0)  # TT right-hand side

x, info = tt.amen(A, b, residual_tol=1e-8, symmetric=True, return_info=True)
print(info.stop_reason, info.residuals[-1])
```

Every operation is exact and stays in the format — nothing is expanded to a
full tensor. Ranks grow as trains are combined, so rounding is part of the
algorithm rather than an optimization; `truncate` and `compress_svd` bound the
error of what they discard.

## Documentation

The `docs/` directory holds the Sphinx sources: a user guide that builds up
tensor networks, the tensor train format and the variational solvers from the
beginning, plus a full API reference.

```bash
python -m pip install -e '.[docs]'
python -m sphinx -b html docs docs/_build/html
```

Open `docs/_build/html/index.html`. A PDF is available through the LaTeX
builder if a TeX distribution is installed:

```bash
python -m sphinx -b latex docs docs/_build/latex && make -C docs/_build/latex
```

## Project structure

```
ttnumpy/
  __init__.py       public API: container, constructors, solvers
  tensor_train.py   the TensorTrain container and its constructors
  solvers/          ALS, MALS and AMEn over a shared core
  utils.py          norms, orthonormalization, SVD truncation
  logger_config.py  application-facing logging setup
  test/             the test suite
docs/               Sphinx sources
```

`ttnumpy.tensor_train` contains the `TensorTrain` container, the constructors
`tt_svd`, `ones`, `zeros`, `eye`, `rand` and `canonical_tt`, and the
normalization, truncation, arithmetic and quantization methods.

`ttnumpy.solvers` contains the iterative solvers: one module per method
(`als`, `mals`, `amen`) over a shared `common` module holding the
environments, the local solves and the stopping machinery. All three are
re-exported from the top level, so `tt.mals(...)` reaches them directly.

`ttnumpy.utils` contains lower-level helpers for tensor norms,
orthonormalization and SVD-based truncation.

## Benchmarks

The `benchmarks/` directory holds a runner that measures the tensor-train
solvers against classical sparse baselines on the Poisson equation
$-\Delta u = f$ on the unit square, discretized with finite differences and
solved on a $2^n \times 2^n$ grid.

### Running

```bash
python -m benchmarks.poisson --sizes 5 6 --output-dir benchmark-results
```

Useful options — `--help` lists them all:

| Option | Meaning |
| --- | --- |
| `--sizes N [N ...]` | QTT depths per axis to run; the grid is $2^n \times 2^n$ |
| `-n N` | A single depth instead of a suite |
| `--tolerance` | Target relative residual for every iterative solver |
| `--mode-size` | Tensorization of the TT operator; 2 is QTT, $2^k$ fuses $k$ cores |
| `--max-rank`, `--max-sweeps`, `--kickrank` | Bond-dimension cap and TT solver limits |
| `--no-reference` | Skip the sparse direct reference solve; error columns become `n/a` |
| `--output-dir` | Where the Markdown and JSON reports are written |

### What it reports

One column per backend — the two TT solvers, SciPy CG, SciPy CG with PyAMG
preconditioning, and PETSc CG with GAMG — and one block of rows per grid size:

The five rows are:

- **real time (s)** — wall-clock time for that backend's solve.
- **err vs exact** — relative error against the sparse direct solution, so the
  accuracy actually achieved rather than the accuracy requested.
- **rel residual** — relative residual $\|Ax - b\| / \|b\|$, the quantity the
  iterative solvers stop on.
- **status** — how the solve ended. The TT solvers report `converged`,
  `stagnated` or `max_sweeps`; the classical ones report `converged` or
  `cg info=<code>`. A backend that exceeds the 600 s per-solve budget is
  reported as `timeout>600s`, and one that raises is reported as
  `failed: <ExceptionType>`, so a single bad backend never aborts the suite.
- **notes** — what the backend did to get there: sweeps and the largest bond
  dimension reached for the TT solvers, iteration count for the classical ones.

## Development

The test suite runs with `pytest` from the repository root:

```bash
pytest
```

Formatting, linting and the documentation build:

```bash
black .
pylint $(git ls-files '*.py')
python -m sphinx -b html -W docs docs/_build/html
```

The bundled `pytest.ini` enables verbose output, short tracebacks and a filter
for the expected SciPy linear-algebra warning used in the tests.

## Logging

TTNumPy follows the usual library convention: it attaches no handlers and stays
quiet until an application asks for output. Call `setup_logging` once from a
script or CLI:

```python
from ttnumpy.logger_config import setup_logging

setup_logging("DEBUG")
```

This configures the `ttnumpy` logger rather than the root logger, so raising
its verbosity never makes third-party libraries verbose. Set
`TTNUMPY_LOG_LEVEL` to pick the level without touching the code:

```bash
TTNUMPY_LOG_LEVEL=DEBUG python my_script.py
```

Applications that already configure logging themselves should skip
`setup_logging` and treat TTNumPy as any other library logger:

```python
logging.getLogger("ttnumpy").setLevel(logging.DEBUG)
```

The package uses two loggers, named by the constants in
`ttnumpy.logger_config`:

- `ttnumpy` (`PACKAGE_LOGGER_NAME`) — everything the library emits.
- `ttnumpy-test` (`TEST_LOGGER_NAME`) — test code. The hyphen matters: logger
  names split on dots, so `ttnumpy-test` is a separate root and raising the
  library's verbosity never raises the tests'.

Sharing one logger name does not lose module identity. A filter stamps every
record with the dotted path of the file that emitted it, so a line still reads:

```
[ttnumpy.solvers.common] INFO MALS stopped after 3 sweep(s) (converged), relative residual 3.4e-07
```

The path is available to any formatter as `%(module_path)s`. Under pytest, do
not call `setup_logging`; use `pytest --log-cli-level=DEBUG` instead.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the
development setup, coding conventions and pull request workflow.

## License

TTNumPy is distributed under the GNU Affero General Public License v3.0.
See [LICENSE](LICENSE) for the full text.
