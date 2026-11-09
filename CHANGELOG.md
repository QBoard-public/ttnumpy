# Changelog

All notable changes to TTNumPy are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html); before 1.0.0 a
minor release may change the API.

## [Unreleased]

### Added

- `TensorTrain.matricize`: the entries of a TT as a vector, of a TTM as a
  matrix with the row modes as rows and the column modes as columns. It works
  for any number of cores when the result fits in memory.

### Changed

- Rounding and building trains with small cores take less time. The
  constructor validates the cores in one pass. `TensorTrain.truncate` copies
  the train once instead of three times. `utils.truncated_bond` and the sweep
  helpers in `utils` make fewer NumPy calls. On trains of 13 binary cores with
  bonds 8–24, truncation takes 8–27% less time and the AMEn and MALS solvers
  take about 8% less.
- `utils.right_orthonormalize` takes a QR factorization of the transposed
  unfolding instead of an RQ factorization. Results of the operations that
  right-orthonormalize a train differ at round-off level: `orthonormalize("right")`,
  `truncate` with the orthogonality centre in the left half, `tt_norm` of a
  train without one, `tt_to_qtt` and the solvers, whose results differ within
  the solver tolerance.

### Fixed

- `TensorTrain.transpose` no longer marks a train as a TTM when every column
  mode stays 1, as in `x.transpose(cores_ind=[])`. The type now follows the
  rule of the constructor, so it no longer changes on `copy()`.

## [0.2.1] - 2026-11-05

### Fixed

- `TensorTrain.truncate` and `compress_svd` (through `utils.left_svd` and
  `utils.right_svd`) no longer raise `LinAlgError` when LAPACK `gesdd` does not
  converge on an unfolding: that unfolding is decomposed again with `gesvd`
  while a warning is logged. If the unfolding has NaN or infinite entries, a
  `ValueError` says so instead.
- `TensorTrain.get_full_tensor` no longer fails on trains of 32 or more cores
  whose result fits in a NumPy array. Inner products such as
  `x.transpose(conjugate=True).dot(y).get_full_tensor()` work for any number of
  cores.
- `TensorTrain.truncate` and `compress_svd` no longer emit a `RuntimeWarning`
  when the product of the bonds overflows int64. The compression ratio in the
  debug log is computed only at DEBUG level, from per-bond ratios.
- `amen` no longer returns bonds above the largest rank the train can have
  there, the product of the modes on the smaller side. The enrichment of the
  last sweep left such bonds next to the first core; they are now reduced
  exactly, so the solution is unchanged.
- The AMEn section of the user guide no longer says that bonds may exceed
  `max_rank`; the enrichment never exceeds it.

## [0.2.0] - 2026-11-02

First release on PyPI. Changes are relative to 0.1.0, which was installed from
the source repository.

### Added

- Tensor cross interpolation: `tci`, `add_global_pivots`, `recompress`,
  `TensorCI`, `TCIInfo` and `BlackBox`, with the building blocks in
  `ttnumpy.interpolation`.
- A user guide chapter on tensor cross interpolation and five worked examples.
- Cost sections for the ALS, MALS and AMEn solvers.
- `ttnumpy.__version__`.
- `CITATION.cff` with citation metadata.

### Changed

- **Breaking:** `TensorTrain.add` (and `x + y`, `x - y`) no longer rounds the
  result, so ranks add. Call `.truncate()` or pass `truncate=True`.
- **Breaking:** Python 3.11 or newer is required.
- **Breaking:** NumPy 2.3.1 or newer is required (was 2.1.0).
- `TensorTrain.truncate`: the default `error` is `0.0` (was `1e-15`).
- `tt_svd`: the default `error` is `1e-12` (was `0.0`).
- Singular values at or below `n * eps * sigma_1` count as zero when a rank is
  chosen, where `n` is the number of singular values and `eps` the machine
  epsilon of the data's precision. Rounding noise no longer adds to the ranks.
- Tests are no longer installed with the package.
- Developer tools are installed through dependency groups:
  `pip install -e . --group dev` (pip 25.1 or newer).

### Removed

- The `dev`, `docs` and `benchmark` extras. The dependency groups of the same
  names replace them.

### Fixed

- Adding complex tensor-train matrices no longer discards the imaginary part.
