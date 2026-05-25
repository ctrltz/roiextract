From extraction pipeline to spatial filter
==========================================

Mapping between MNE-Python functions and ROIextract classes:

+-------------------------------------------------------------------+-------------------------------------------------------------------+
| **Function in MNE-Python**                                        | **Corresponding pipeline step in ROIextract**                     |
+-------------------------------------------------------------------+-------------------------------------------------------------------+
| :func:`~mne.minimum_norm.apply_inverse_raw`                       | :class:`~roiextract.pipeline.Inverse`                             |
+-------------------------------------------------------------------+-------------------------------------------------------------------+
| :func:`~mne.extract_label_time_course` with ``mode="mean"``       | :class:`~roiextract.pipeline.MeanAggregation`                     |
+-------------------------------------------------------------------+-------------------------------------------------------------------+
| :func:`~mne.extract_label_time_course` with ``mode="mean_flip"``  | :class:`~roiextract.pipeline.MeanAggregation` with ``flip=True``  |
+-------------------------------------------------------------------+-------------------------------------------------------------------+

Additional options available in ROIextract:

* :class:`~roiextract.pipeline.CentroidAggregation` - centroid-based aggregation.
