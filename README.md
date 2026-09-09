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
NumPy (>= 2.1) and SciPy (>= 1.15).

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
development setup, coding conventions and merge request workflow.

## License

TTNumPy is distributed under the GNU Affero General Public License v3.0.
See [LICENSE](LICENSE) for the full text.
