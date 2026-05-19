import mne

from scipy import sparse

from roiextract.pipeline.pipeline import PipelineStep, StepType
from roiextract.utils import get_label_mask


class MeanAggregation(PipelineStep):
    kind = StepType.ROIAggregation

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

    @property
    def weights(self):
        self._check_if_prepared()
        return self._weights

    @property
    def row_names(self):
        self._check_if_prepared()
        return self.names
