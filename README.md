# ROIextract

Optimization of extraction of ROI time series based on the cross-talk function (CTF) or source reconstruction of spatial patterns (REC). **Work in progress!**

## Usage

Obtain a spatial filter that optimizes CTF properties:

```python
from roiextract import ctf_optimize_label

sf = ctf_optimize_label(fwd, label, template, alpha)

sf, props = ctf_optimize_label(fwd, label, template, alpha, quantify=True)

sf = ctf_optimize_label(fwd, label, template, alpha='auto', threshold=0.95)
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
