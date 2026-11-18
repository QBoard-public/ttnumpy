# Changelog

All notable changes to TTNumPy are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html); before 1.0.0 a
minor release may change the API.

## [0.3.0] - 2026-11-18

### Added

- `TensorTrain.matricize`: the entries of a TT as a vector, of a TTM as a
  matrix with the row modes as rows and the column modes as columns. It works
  for any number of cores when the result fits in memory.
- `TensorTrain.add`, `multiply` and `dot` take `error` and `max_rank` together
  with `truncate=True` and round the result with them, as `truncate(error,
  max_rank)` does. Given without `truncate=True`, they raise `ValueError`.
- `TensorTrain.truncate` takes `method="randomized"` with `seed` and `options`.
  It orthonormalizes the train on a random sketch with bonds `max_rank` plus an
  oversampling of 10, then truncates with the SVD sweep as before. It needs
  `max_rank`, and `error` bounds the SVD sweep only. `add`, `multiply` and
  `dot` take the same arguments together with `truncate=True`. On trains of
  18 binary cores with bonds 96–685, rounded to bonds 32–80, it takes 9–60%
  less time than `"direct"`. The error is 1.0–1.3 times that of `"direct"` in
  the median over 20 seeds and at most 2.6 times. On trains with bonds close
  to `max_rank + 10` (24 for 22, 48 for 42) it takes 30–38% more time.
- `utils.left_randomized_orthonormalize` and
  `utils.right_randomized_orthonormalize`: the sweeps behind the randomized
  method.
- `TensorTrain.multiply` of two tensor trains with `truncate=True` and
  `method="randomized"` no longer builds the product with bonds `r * s`. It
  takes the left-orthonormal cores from the new `utils.randomized_hadamard`,
  the HaTT-1 algorithm of Sun, Huang, Xiao and Yang (arXiv:2410.04385). For
  factors with bonds 16–64 on 18 binary cores, rounded to bonds 32–64, it
  takes 80–91% less time than building the product and rounding it with
  `"direct"`. The error is 1.05–1.42 times that of `"direct"` in the median
  over 20 seeds and at most 1.7 times.
- `benchmarks.arithmetic`: a benchmark of the arithmetic of tensor trains. Its
  accuracy table compares products, sums, an operator applied to a vector and
  a polynomial of a vector on QTT vectors of up to $2^{40}$ entries with
  NumPy: memory of both forms, error of the train. Its speed table times the
  direct and the randomized rounding on the same workloads. It writes
  Markdown and JSON reports like the Poisson benchmark.

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
- AMEn computes the cores of its residual train with three matrix products
  instead of one `numpy.einsum` call. When the solution bond was between about
  the operator bond and twice it, `einsum` ran one loop over all eight indices.
  A sweep of AMEn takes 33–44% less time on the 2D Poisson problem of the
  benchmarks (operator bond 6, grids from 128 × 128 to 512 × 512) and 52–88%
  less on QTT finite-element systems with operator bonds 8–63. Solutions
  differ within the solver tolerance, which can change the number of sweeps
  by one.

### Fixed

- `TensorTrain.transpose` no longer marks a train as a TTM when every column
  mode stays 1, as in `x.transpose(cores_ind=[])`. The type now follows the
  rule of the constructor, so it no longer changes on `copy()`.
- `TensorTrain.truncate` on a train of two cores now rounds as on longer
  trains. Its SVD sweep started at the core away from the orthogonality
  centre: `error` removed no bond, while `max_rank` gave errors up to 15 times
  the smallest possible one. The same held for `add`, `multiply` and `dot`
  with `truncate=True`, for `method="randomized"` and for `qtt_to_tt` with two
  groups. `compress_svd(canonical="auto")` now starts its sweep at the end
  nearer to `orth_center`. Results change only for trains of two cores, which
  now end with the centre on the first core as longer trains do, and for
  `compress_svd` on an even number of cores with `orth_center` in the middle.
- A NumPy scalar on the left of a train, as in `np.float64(2.0) * train`, now
  scales the train like a Python scalar. NumPy used to take over the operator
  and raise `ValueError`. For a train of rank 1 with equal modes it returned
  an array of the scaled cores. Other operators with a NumPy object on the
  left, such as `np.float64(2.0) + train`, raise `TypeError` instead of
  `ValueError`.
- `TensorTrain.qtt_to_tt` merges a group of 32 or more QTT cores into one
  core. It raised `ValueError` on NumPy's limit of 64 axes, which a group that
  fits in memory reaches only with modes of size 1. The group is now
  contracted as a matrix, as `matricize` does. Results of the groupings that
  worked before are unchanged.

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
