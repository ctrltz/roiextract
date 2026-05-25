import logging
import mne
import numpy as np
import typing as T

from scipy import sparse

from roiextract.pipeline.step import PipelineStep
from roiextract.utils import get_label_mask, vertno_to_index


class MeanAggregation(PipelineStep):
    """
    Averaging-based aggregation of reconstructed source time courses within
    the ROI. Optionally, a sign flip can be applied before averaging to
    reduce potential cancellation of activity. This pipeline step corresponds to
    the :func:`mne.extract_label_time_course` function with ``mode="mean"`` or
    ``mode="mean_flip"``.

    Parameters
    ----------
    flip : bool, default=False
        Whether to apply a sign flip before averaging. Sign flip is determined
        based on the singular value decomposition of the leadfield, as
        performed in :func:`mne.label_sign_flip`.
    """

    def __init__(self, flip=False) -> None:
        super().__init__()
        self.flip: bool = flip

        self.labels: list[mne.Label] = []
        self.src: mne.SourceSpaces | None = None
        self.names: list[str] = []
        self.prepared: bool = False
        self._weights: sparse.csr_matrix = sparse.csr_matrix((0, 0))

    def __repr__(self) -> str:
        return "MeanFlipAggregation" if self.flip else "MeanAggregation"

    def fit(  # type: ignore[override]
        self,
        data: mne.SourceEstimate,
        src: mne.SourceSpaces,
        labels: mne.Label | list[mne.Label],
    ) -> "MeanAggregation":
        """
        Fit the aggregation step to the provided data, source space, and labels.

        Parameters
        ----------
        data : SourceEstimate
            The source estimate containing the reconstructed source time courses.
        src : SourceSpaces
            The definition of the considered source space for inverse modeling.
        labels : Label | list of Label
            The label or list of labels defining the ROIs for which time courses
            should be extracted.

        Returns
        -------
        self : MeanAggregation
            The fitted aggregation step.
        """
        self.labels = labels if isinstance(labels, list) else [labels]
        self.src = src
        self.names = [label.name for label in self.labels]

        n_sources = data.shape[0]
        n_labels = len(self.labels)
        weights = sparse.lil_matrix((n_labels, n_sources))

        for i, label in enumerate(self.labels):
            mask = get_label_mask(label, src)
            weights[i, mask] = (
                1 if not self.flip else mne.label_sign_flip(label, src)
            ) / mask.sum()

        self._weights = weights.tocsr()

        self.prepared = True
        return self

    def transform(self, data: mne.SourceEstimate) -> np.ndarray:
        """
        Apply the fitted aggregation to the provided data. The applied transformation
        corresponds to the :func:`mne.extract_label_time_course` function
        with ``mode="mean"`` or ``mode="mean_flip"`` for ``flip=False`` and
        ``flip=True``, respectively.

        Parameters
        ----------
        data : SourceEstimate
            The source estimate containing the reconstructed source time courses.

        Returns
        -------
        label_tc : array, shape (n_labels, n_times)
            The extracted time courses for each label.
        """
        self._check_if_prepared()
        return mne.extract_label_time_course(
            data, self.labels, src=self.src, mode="mean_flip" if self.flip else "mean"
        )

    def fit_transform(  # type: ignore[override]
        self,
        data: mne.SourceEstimate,
        src: mne.SourceSpaces,
        labels: mne.Label | list[mne.Label],
    ) -> np.ndarray:
        """
        Fit the aggregation step to the provided data, source space, and labels,
        and apply the aggregation to extract the ROI time courses. See
        :meth:`fit()` and :meth:`transform()` for details on the parameters and
        return values, respectively.
        """
        self.fit(data, src, labels)
        return self.transform(data)

    def _request_args(
        self,
        src: mne.SourceSpaces,
        labels: mne.Label | list[mne.Label],
        subject: str | None = None,
        subjects_dir: str | None = None,
        **kwargs: T.Any,
    ) -> dict:
        return dict(src=src, labels=labels)

    def get_weights(self) -> sparse.csr_matrix:
        """
        Weight matrix corresponding to the resulting aggregation transformation.

        Returns
        -------
        weights : array
            The weight matrix.
        """
        self._check_if_prepared()
        return self._weights

    def get_names(self) -> list[str]:
        """
        Label names are used as names for rows of the weight matrix.

        Returns
        -------
        row_names : list of str
            Names for rows of the weight matrix.
        """
        self._check_if_prepared()
        return self.names

    def get_params(self) -> dict:
        """
        Get the single ``flip`` parameter of the aggregation step as a dictionary.

        Returns
        -------
        params : dict
            The parameters of the aggregation step.
        """
        return dict(flip=self.flip)


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

    def __init__(self, surf="sphere") -> None:
        super().__init__()
        self.surf: str = surf
        self.labels: list[mne.Label] = []
        self.src: mne.SourceSpaces | None = None
        self.names: list[str] = []
        self._weights: sparse.csr_matrix = sparse.csr_matrix((0, 0))
        self._indices: np.ndarray = np.array([], dtype=int)

    def __repr__(self) -> str:
        return "CentroidAggregation"

    def fit(  # type: ignore[override]
        self,
        data: mne.SourceEstimate,
        src: mne.SourceSpaces,
        labels: mne.Label | list[mne.Label],
        subject: str | None = None,
        subjects_dir: str | None = None,
    ) -> "CentroidAggregation":
        """
        Fit the aggregation step to the provided data, source space, and labels.

        Parameters
        ----------
        data : SourceEstimate
            The source estimate containing the reconstructed source time courses.
        src : SourceSpaces
            The definition of the considered source space for inverse modeling.
        labels : Label | list of Label
            The label or list of labels defining the ROIs for which time courses
            should be extracted.
        subject : str | None
            The subject name. If ``None``, it will be inferred from the source space.
        subjects_dir : str | None
            The directory containing the subjects' MRI data. If ``None``, it will be
            inferred from the environment by MNE-Python. Set the path explicitly in
            case of errors.

        Returns
        -------
        self : CentroidAggregation
            The fitted aggregation step.
        """
        self.labels = labels if isinstance(labels, list) else [labels]
        self.src = src
        self.names = [label.name for label in self.labels]

        n_sources = data.shape[0]
        n_labels = len(self.labels)
        weights = sparse.lil_matrix((n_labels, n_sources))
        self._indices = np.zeros(n_labels, dtype=int)

        if subject is None:
            logging.warning(
                "Subject name is not provided explicitly, "
                "attempting to infer from source space."
            )
            subject = src[0]["subject_his_id"]

        for i, label in enumerate(self.labels):
            centroid_vertno = label.center_of_mass(
                subject=subject,
                restrict_vertices=src,
                subjects_dir=subjects_dir,
                surf=self.surf,
            )
            centroid_index = vertno_to_index(src, label.hemi, centroid_vertno)
            self._indices[i] = centroid_index

            weights[i, centroid_index] = 1

        self._weights = weights.tocsr()

        self.prepared = True
        return self

    def transform(self, data: mne.SourceEstimate) -> np.ndarray:
        """
        Apply centroid-based aggregation to the provided data.

        Parameters
        ----------
        data : SourceEstimate
            The source estimate containing the reconstructed source time courses.

        Returns
        -------
        label_tc : array, shape (n_labels, n_times)
            The extracted time courses for each label.
        """
        self._check_if_prepared()
        return data.data[self._indices, :]

    def fit_transform(  # type: ignore[override]
        self,
        data: mne.SourceEstimate,
        src: mne.SourceSpaces,
        labels: mne.Label | list[mne.Label],
        subject: str | None = None,
        subjects_dir: str | None = None,
    ) -> np.ndarray:
        """
        Fit the aggregation step to the provided data, source space, and labels,
        and apply the aggregation to extract the ROI time courses. See
        :meth:`fit()` and :meth:`transform()` for details on the parameters and
        return values, respectively.
        """
        self.fit(data, src, labels, subject, subjects_dir)
        return self.transform(data)

    def _request_args(
        self,
        src: mne.SourceSpaces,
        labels: mne.Label | list[mne.Label],
        subject: str | None = None,
        subjects_dir: str | None = None,
        **kwargs: T.Any,
    ) -> dict[str, T.Any]:
        return dict(src=src, labels=labels, subject=subject, subjects_dir=subjects_dir)

    def get_weights(self) -> sparse.csr_matrix:
        """
        The weight matrix corresponding to the resulting aggregation transformation.

        Returns
        -------
        weights : array
            The weight matrix that contains one non-zero entry per row corresponding
            to the selected centroid source for each label.
        """
        self._check_if_prepared()
        return self._weights

    def get_names(self) -> list[str] | None:
        """
        Label names are used as names for rows of the weight matrix.

        Returns
        -------
        names : list of str
            The names of the rows / labels.
        """
        self._check_if_prepared()
        return self.names

    def get_params(self) -> dict[str, T.Any]:
        """
        Get the single ``surf`` parameter of the aggregation step as a dictionary.

        Returns
        -------
        params : dict
            The parameters of the aggregation step.
        """
        return dict(surf=self.surf)
