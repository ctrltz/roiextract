# ROIextract

Optimization of extraction of ROI time series based on the cross-talk function (CTF) or source reconstruction of spatial patterns (REC). **Work in progress!**

## Usage

Obtain a spatial filter that optimize CTF properties:

```python
from roiextract import ctf_optimize_label

sf = ctf_optimize_label(fwd, label, template, alpha)

sf, props = ctf_optimize_label(fwd, label, template, alpha, quantify=True)

sf = ctf_optimize_label(fwd, label, template, alpha='auto', threshold=0.95)
```

Inspect or apply the filter:

```python
sf.plot(info)

sf.apply(data)
```

Estimate the CTF for the filter:

```python
ctf = sf.get_ctf_fwd(fwd)  # ctf is an instance of mne.SourceEstimate
```

Inspect the CTF:

```python
ctf.plot()
```
