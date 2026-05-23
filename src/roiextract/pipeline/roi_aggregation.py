import logging
import mne
import numpy as np

from scipy import sparse

from roiextract.pipeline.step import PipelineStep
from roiextract.utils import get_label_mask, vertno_to_index


class MeanAggregation(PipelineStep):
    """
    Averaging-based aggregation of reconstructed source time courses within
    the ROI. Optionally, a sign flip can be applied before averaging to
    reduce potential cancellation of activity.

    Parameters
    ----------
    flip : bool, default=False
        Whether to apply a sign flip before averaging. Sign flip is determined
        based on the singular value decomposition of the leadfield, as
        performed in :func:`mne.label_sign_flip`.
    """

    def __init__(self, flip=False):
        super().__init__()
        self.flip = flip

        self.labels = None
        self.src = None
        self.names = None
        self.prepared = False
        self._weights = None

    def __repr__(self):
        return "MeanFlipAggregation" if self.flip else "MeanAggregation"

    def fit(self, data, src, labels):
        self.labels = labels
        self.src = src
        self.names = [label.name for label in labels]

        n_sources = data.shape[0]
        n_labels = len(labels)
        self._weights = sparse.lil_matrix((n_labels, n_sources))

        for i, label in enumerate(labels):
            mask = get_label_mask(label, src)
            self._weights[i, mask] = (
                1 if not self.flip else mne.label_sign_flip(label, src)
            ) / mask.sum()

        self._weights = self._weights.tocsr()

        self.prepared = True
        return self

    def transform(self, data):
        self._check_if_prepared()
        return mne.extract_label_time_course(
            data, self.labels, src=self.src, mode="mean_flip" if self.flip else "mean"
        )

    def fit_transform(self, data, src, labels):
        self.fit(data, src, labels)
        return self.transform(data)

    def request_args(self, src, labels, subject=None, subjects_dir=None, **kwargs):
        return dict(src=src, labels=labels)

    @property
    def weights(self):
        self._check_if_prepared()
        return self._weights

    @property
    def row_names(self):
        self._check_if_prepared()
        return self.names


class CentroidAggregation(PipelineStep):
    """
    Centroid-based aggregation of reconstructed source time courses within the
    ROI. The time course of the source that is the closest to the center of mass
    of the ROI is selected as the representative time course of the ROI.

    Parameters
    ----------
    surf : str, default="sphere"
        The surface to use for computing the center of mass. The provided value
        is forwarded to :meth:`mne.Label.center_of_mass` without modification.
    """

    def __init__(self, surf="sphere"):
        super().__init__()
        self.surf = surf
        self._weights = None
        self._indices = None
        self.labels = None
        self.src = None
        self.names = None

    def __repr__(self):
        return "CentroidAggregation"

    def fit(self, data, src, labels, subject=None, subjects_dir=None):
        self.labels = labels
        self.src = src
        self.names = [label.name for label in labels]

        n_sources = data.shape[0]
        n_labels = len(labels)
        self._weights = sparse.lil_matrix((n_labels, n_sources))
        self._indices = np.zeros(n_labels, dtype=int)

        if subject is None:
            logging.warning(
                "Subject name is not provided explicitly, "
                "attempting to infer from source space."
            )
            subject = src[0]["subject_his_id"]

        for i, label in enumerate(labels):
            centroid_vertno = label.center_of_mass(
                subject=subject,
                restrict_vertices=src,
                subjects_dir=subjects_dir,
                surf=self.surf,
            )
            centroid_index = vertno_to_index(src, label.hemi, centroid_vertno)
            self._indices[i] = centroid_index

            self._weights[i, centroid_index] = 1

        self._weights = self._weights.tocsr()

        self.prepared = True
        return self

    def transform(self, data):
        self._check_if_prepared()
        return data.data[self._indices, :]

    def fit_transform(self, data, src, labels, subject=None, subjects_dir=None):
        self.fit(data, src, labels, subject, subjects_dir)
        return self.transform(data)

    def request_args(self, src, labels, subject=None, subjects_dir=None, **kwargs):
        return dict(src=src, labels=labels, subject=subject, subjects_dir=subjects_dir)

    @property
    def weights(self):
        self._check_if_prepared()
        return self._weights

    @property
    def row_names(self):
        self._check_if_prepared()
        return self.names
