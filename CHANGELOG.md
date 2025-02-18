# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Version 0.0.4 (Unreleased)

### Added

* Function for calculating the cosine similarity between spatial filters ([#17](https://github.com/ctrltz/roiextract/pull/17))

### Removed

* Dropped the support for Python 3.7 and 3.8 ([#17](https://github.com/ctrltz/roiextract/pull/17))

## Version 0.0.3 (2024-05-29)

### Added

* Support for source space mask to consider only a subset of source in all CTF calculations
* Use of several initial guesses for improving the numerical optimization
* Support of the corner case of ROIs consisting of a single source
* First sketches of documentation, still very much work in progress

### Changed

* Fields of the SpatialFilter now have higher flexibility in describing the method for obtaining the spatial filter

## Version 0.0.2 (2024-01-25)

### Added

* Estimation of CTF homogeneity
* Numerical solution for optimizing a combination of CTF ratio and homogeneity
* Tests to cover the basic functionality
* Binary search for suggesting lambda_ based on the desired properties
* SpatialFilter class for storing the result of the optimization

### Changed

* Re-structured the package, split into several files preparing for future extensions (free orientations, data-driven method)

## Version 0.0.1 (2023-05-08)

### Added

* Analytic solution for a spatial filter that optimizes a combination of CTF ratio and similarity with a template
* Structured the code as a Python package
