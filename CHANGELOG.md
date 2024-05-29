# Changelog

## 0.0.3 - May 29th, 2024

### Added

* Support for source space mask to consider only a subset of source in all CTF calculations
* Use of several initial guesses for improving the numerical optimization
* Support of the corner case of ROIs consisting of a single source
* First sketches of documentation, still very much work in progress

### Changed

* Fields of the SpatialFilter now have higher flexibility in describing the method for obtaining the spatial filter

## 0.0.2 - January 25th, 2024

### Added

* Estimation of CTF homogeneity
* Numerical solution for optimizing a combination of CTF ratio and homogeneity
* Tests to cover the basic functionality
* Binary search for suggesting lambda_ based on the desired properties
* SpatialFilter class for storing the result of the optimization

### Changed

* Re-structured the package, split into several files preparing for future extensions (free orientations, data-driven method)

## 0.0.1 - May 8th, 2023

### Added

* Analytic solution for a spatial filter that optimizes a combination of CTF ratio and similarity with a template
* Structured the code as a Python package
