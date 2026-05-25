1. Extraction pipelines
=======================

Mapping between MNE-Python functions and ROIextract classes:

+------------------------+-------------------------------------------------------------------------------------------------+-------------------------------------------------------------------+
| **Pipeline step type** | **Function(s) in MNE-Python**                                                                   | **Corresponding pipeline step in ROIextract**                     |
+------------------------+-------------------------------------------------------------------------------------------------+-------------------------------------------------------------------+
| Source reconstruction  | :func:`~mne.minimum_norm.prepare_inverse_operator`, :func:`~mne.minimum_norm.apply_inverse_raw` | :class:`~roiextract.pipeline.Inverse`                             |
+------------------------+-------------------------------------------------------------------------------------------------+-------------------------------------------------------------------+
| ROI aggregation        | :func:`~mne.extract_label_time_course` with ``mode="mean"``                                     | :class:`~roiextract.pipeline.MeanAggregation`                     |
+------------------------+-------------------------------------------------------------------------------------------------+-------------------------------------------------------------------+
| ROI aggregation        | :func:`~mne.extract_label_time_course` with ``mode="mean_flip"``                                | :class:`~roiextract.pipeline.MeanAggregation` with ``flip=True``  |
+------------------------+-------------------------------------------------------------------------------------------------+-------------------------------------------------------------------+

Additional options available in ROIextract:

* :class:`~roiextract.pipeline.CentroidAggregation` - centroid-based aggregation.
