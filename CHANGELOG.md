# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2025-05-17

### Added

- Add RemainingValueTuner to transform discrete trend labels into continuous values expressing trend potential.
- Add smoothing utilities.
- Add label tuner example notebook.

### Changed

- Update documentation to include new features.
- Update README.md to include new features.


## [0.2.1] - 2025-03-30

### Added

- Add automated documentation using `sphinx`
- Add readthedocs.yaml for readthedocs integration

### Changed

- Fixed bibliography in README.md
- Fix docstrings to adhere to sphinx recommendations

## [0.2.0] - 2025-02-19

### Added

- Added `poetry` to the project.

### Changed

- Update and fix dependencies with poetry.
- Restructure modules and imports.
- Rename package to tstrends.
- Update README to render on pypi.
- Update notebooks to comply with new imports.


## [0.1.1] - 2025-02-19

### Changed

- Reorganized module structure and imports.
- Removed unused imports across codebase for cleaner code

### Fixed

- Updated GitHub Actions workflow to:
  - Run tests using `python -m pytest` for proper module resolution
  - Install package in editable mode with `pip install -e .` before running tests


## [0.1.0] - 2025-02-18

### Added

- First beta release.
- Trend labelling algorithms:
  - Binary CTL
  - Ternary CTL
  - Oracle Binary Trend Labeller
  - Oracle Ternary Trend Labeller
- Returns estimation with transaction costs and holding fees.
- Parameter optimization with Bayesian optimization.
- Unit and integration tests.
- Example notebooks