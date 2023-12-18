# ROIextract

Optimization of extraction of ROI time series based on the cross-talk function (CTF) or source reconstruction of spatial patterns (REC). **Work in progress!**

## Usage

```python
from roiextract import ctf_optimize_label

w_opt = ctf_optimize_label(fwd, label, template, alpha)

w_opt, props = ctf_optimize_label(fwd, label, template, alpha, quantify=True)

w_opt = ctf_optimize_label(fwd, label, template, alpha='auto', threshold=0.95)
```