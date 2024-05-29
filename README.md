# ROIextract

Optimization of spatial filters for extraction of ROI time series based on the cross-talk function (CTF) or source reconstruction of spatial patterns (REC). **Work in progress!**

## Background

TODO

## Prerequisites

The toolbox is designed to be compatible with [MNE-Python]() as much as possible, and optimization only requires:

* `fwd: mne.Forward` - the description of the forward model
* `label: mne.Label` - the description of the region of interest (parcel/label)

## Parameters

There are several parameters of the optimization that need to be specified by the user:

* `lambda_` - this parameter allows fine-tuning the cross-talk function according to the demands of the analysis:
  * `lambda_ = 0` prioritizes the CTF ratio, leading to better localization of sources potential contributing to the extracted signal
  * `lambda_ = 1` prioritizes the CTF homogeneity, allowing to force contributions from the sources within the ROI to be more similar to a pre-specified template (see below)
  * values between 0 and 1 lead to a compromise between ratio and homogeneity
  * `lambda = auto` (experimental) - automatically selects the value of `lambda_` to obtain a fraction (controlled with `threshold`) of the maximal CTF ratio or homogeneity (controlled with `criteria`):
    * `threshold` - a number between 0 and 1 controlling the fraction
    * `criteria` - use either ratio (`criteria="ratio"`) or homogeneity (`criteria="homogeneity"`) for automatic suggestion of `lambda_`
* `template` - this parameter allows specifying the desired contributions of sources within ROIs, the following options are available:
  * `mean` - equal values for all sources (homogeneous contribution)
  * custom templates (e.g., gaussian) may be provided directly as an array of weights for all sources within the ROI

## Usage

Obtain a spatial filter that optimizes CTF properties:

```python
from roiextract import ctf_optimize_label

sf = ctf_optimize_label(fwd, label, template, lambda_)

sf, props = ctf_optimize_label(fwd, label, template, lambda_, quantify=True)

sf = ctf_optimize_label(fwd, label, template, lambda_='auto', threshold=0.95)
```

Plot the filter as a topomap:

```python
sf.plot(info)
```

Apply it to the data to obtain the time course of activity in the ROI/label:

```python
label_tc = sf.apply(data)
```

Estimate the CTF for the filter:

```python
ctf = sf.get_ctf_fwd(fwd)  # ctf is an instance of mne.SourceEstimate
```

Plot the CTF on the brain surface:

```python
ctf.plot()
```
