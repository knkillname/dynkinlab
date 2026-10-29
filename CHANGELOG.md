# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- The initial typed `dynkinlab` package skeleton, built with Hatchling and targeting Python 3.13.
- Immutable symmetric `QuasiCartanMatrix` values with construction-time validation.
- Exact integer `QuadraticForm` evaluation and exact `Inertia` computation by rational congruence
  reduction, including a public `Inertia.from_matrix` entry point.
- Colab/Jupyter Markdown and HTML rendering for matrices, forms, and inertia, with notebook-wide
  `dynkinlab.display.options` and per-call `DisplayOptions` overrides.
- A pinned Colab compatibility smoke gate for the official CPU runtime image.
- Sphinx documentation, a versioned pre-push hook, and a CI workflow for checks, tests, packaging,
  documentation, and Colab compatibility.
