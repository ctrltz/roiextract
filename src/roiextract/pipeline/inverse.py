import mne
import numpy as np
import typing as T

from mne._fiff.constants import FIFF
from mne.minimum_norm import (
    apply_inverse_raw,
    InverseOperator,
    prepare_inverse_operator,
)

from roiextract.pipeline.step import PipelineStep
from roiextract.pipeline.utils import _get_matrix_from_prepared_inverse_operator


class Inverse(PipelineStep):
    """
    Source reconstruction via an inverse operator. This step wraps the
    :func:`mne.minimum_norm.apply_inverse_raw` function, allowing quick
    access to the weight matrix corresponding to the inverse operator.

    .. note::
        Currently, only fixed source orientations are supported.

    Parameters
    ----------
    inv : InverseOperator
        The inverse operator to be used for source reconstruction.
    method : str
        The name of the reconstruction method to use. Supported methods include
        ``"MNE"``, ``"dSPM"``, ``"sLORETA"``, and ``"eLORETA"``.
    lambda2 : float
        The regularization parameter for the inverse operator.
    nave : int, default=1
        Number of averages used to regularize the solution. Set to 1 on raw data.
    """

    def __init__(
        self, inv: InverseOperator, method: str, lambda2: float, nave: int = 1
    ) -> None:
        super().__init__()
        if inv["source_ori"] != FIFF.FIFFV_MNE_FIXED_ORI:
            raise ValueError("Only fixed source orientations are supported")

        self._inv_op: InverseOperator = inv.copy()
        self.method: str = method
        self.lambda2: float = lambda2
        self.nave: int = nave
        self.apply_fun: T.Callable | None = None

    def __repr__(self) -> str:
        return f"Inverse<{self.method}>"

    def fit(self, data: mne.io.BaseRaw) -> "Inverse":  # type: ignore[override]
        """
        Fit the inverse operator to the provided data.

        Parameters
        ----------
        data : Raw
            The raw data to fit the inverse operator on.

        Returns
        -------
        self : Inverse
            The fitted inverse operator.
        """
        if not isinstance(data, mne.io.BaseRaw):
            raise ValueError("Only mne.io.Raw objects are supported")

        self.apply_fun = apply_inverse_raw
        self._inv_op = prepare_inverse_operator(
            orig=self._inv_op, nave=self.nave, lambda2=self.lambda2, method=self.method
        )
        self._weights = _get_matrix_from_prepared_inverse_operator(
            self._inv_op, self.method, self.lambda2
        )
        self.prepared = True

        return self

    def transform(self, data: mne.io.BaseRaw) -> mne.SourceEstimate:
        """
        Apply the fitted inverse operator to the provided data.

        Parameters
        ----------
        data : Raw
            The raw data to apply the inverse operator on.

        Returns
        -------
        stc : SourceEstimate
            The source estimate obtained by applying the inverse operator.
        """
        self._check_if_prepared()
        if self.apply_fun is None:
            raise RuntimeError("The apply function has not been set. Call fit() first.")

        return self.apply_fun(
            data, self._inv_op, method=self.method, lambda2=self.lambda2, prepared=True
        )

    def fit_transform(self, data: mne.io.BaseRaw) -> mne.SourceEstimate:  # type: ignore[override]
        """
        Fit the inverse operator to the provided data and then apply it.

        Parameters
        ----------
        data : Raw
            The raw data to fit and apply the inverse operator on.

        Returns
        -------
        stc : SourceEstimate
            The source estimate obtained by applying the inverse operator.
        """
        return self.fit(data).transform(data)

    def get_weights(self) -> np.ndarray:
        """
        The weight matrix corresponding to the fitted inverse operator.

        Returns
        -------
        weights : array, shape (n_sources, n_sensors)
            The weight matrix of the inverse operator.
        """
        self._check_if_prepared()
        return self._weights

    def get_params(self) -> dict[str, T.Any]:
        return dict(method=self.method, lambda2=self.lambda2, nave=self.nave)
