# Changelog

All notable changes to TTNumPy are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html); before 1.0.0 a
minor release may change the API.

## [0.2.0] - unreleased

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
