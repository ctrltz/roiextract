import mne
import numpy as np
import typing as T

from mne._fiff.constants import FIFF
from mne.beamformer import make_lcmv, apply_lcmv_raw, Beamformer
from mne.minimum_norm import (
    apply_inverse_raw,
    InverseOperator,
    prepare_inverse_operator,
)

from roiextract.pipeline.step import PipelineStep
from roiextract.pipeline.utils import (
    _get_matrix_from_prepared_inverse_operator,
    _get_matrix_from_lcmv_filters,
)


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

        self._inv_orig: InverseOperator = inv.copy()
        self._inv_prepared: InverseOperator = inv.copy()
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
        self._inv_prepared = prepare_inverse_operator(
            orig=self._inv_orig,
            nave=self.nave,
            lambda2=self.lambda2,
            method=self.method,
        )
        self._weights = _get_matrix_from_prepared_inverse_operator(
            self._inv_prepared, self.method, self.lambda2
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
            data,
            self._inv_prepared,
            method=self.method,
            lambda2=self.lambda2,
            prepared=True,
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


class LCMVBeamformer(PipelineStep):
    """
    Source reconstruction via an LCMV beamformer. This step wraps the
    :func:`~mne.beamformer.make_lcmv` and
    :func:`~mne.beamformer.apply_lcmv_raw` functions, allowing quick access
    to the LCMV beamformer weights.

    Parameters
    ----------
    fwd : Forward
        The forward solution to be used for source reconstruction.
    reg : float
        Regularization parameter for the LCMV beamformer.
    weight_norm : str
        The weight normalization method to use. Supported methods include
        ``None`` (corresponding to a unit-gain beamformer),
        ``"unit-noise-gain"``, ``"unit-noise-gain-invariant"`` (default),
        and ``"nai"``.
    cov_tstep : float
        The time step for computing the data covariance matrix. It is used to
        compute the covariance matrix from the raw data using
        :func:`mne.compute_raw_covariance` if no data covariance is provided.
    """

    def __init__(
        self,
        fwd: mne.Forward,
        reg: float = 0.05,
        weight_norm: str = "unit-noise-gain-invariant",
        cov_tstep: float = 2.0,
    ) -> None:
        super().__init__()
        self.fwd = fwd
        self.reg = reg
        self.weight_norm = weight_norm
        self.cov_tstep = cov_tstep

        self.filters: Beamformer | None = None
        self._weights: np.ndarray | None = None
        self.apply_fun: T.Callable | None = None

    def __repr__(self) -> str:
        return "LCMVBeamformer"

    def _request_args(
        self,
        src: mne.SourceSpaces,
        labels: mne.Label | list[mne.Label],
        subject: str | None = None,
        subjects_dir: str | None = None,
        **kwargs,
    ):
        """
        Allow providing custom data and noise covariance matrices via
        keyword arguments of the fit method.
        """
        data_cov = kwargs.get("data_cov", None)
        noise_cov = kwargs.get("noise_cov", None)
        return dict(data_cov=data_cov, noise_cov=noise_cov)

    def fit(  # type: ignore[override]
        self,
        data: mne.io.BaseRaw,
        data_cov: mne.Covariance | None = None,
        noise_cov: mne.Covariance | None = None,
    ) -> "LCMVBeamformer":
        """
        Fit the LCMV beamformer to the provided data, optionally using custom
        data and noise covariance matrices. If the custom matrices are not
        provided, by default, the data covariance is computed from the raw data
        using :func:`mne.compute_raw_covariance`, and the noise covariance is
        generated using :func:`mne.make_ad_hoc_cov` with a standard deviation
        of 1.0.

        Parameters
        ----------
        data : Raw
            The raw data to fit the LCMV beamformer on.
        data_cov : Covariance | None, optional
            The data covariance matrix. If None, it will be computed from
            the raw data.
        noise_cov : Covariance | None, optional
            The noise covariance matrix. If None, an ad-hoc covariance matrix
            will be created.

        Returns
        -------
        self : LCMVBeamformer
            The fitted LCMV beamformer.
        """
        if not isinstance(data, mne.io.BaseRaw):
            raise ValueError("Only mne.io.Raw objects are supported")

        if data_cov is None:
            data_cov = mne.compute_raw_covariance(data, tstep=self.cov_tstep)

        if noise_cov is None:
            noise_cov = mne.make_ad_hoc_cov(data.info, std=1.0)

        self.apply_fun = apply_lcmv_raw
        self.filters = make_lcmv(
            info=data.info,
            forward=self.fwd,
            data_cov=data_cov,
            reg=self.reg,
            noise_cov=noise_cov,
            weight_norm=self.weight_norm,
        )
        self._weights = _get_matrix_from_lcmv_filters(data.info, self.filters)
        self.prepared = True
        return self

    def transform(self, data: mne.io.BaseRaw) -> mne.SourceEstimate:
        """
        Apply the fitted LCMV beamformer to the provided data.

        Parameters
        ----------
        data : Raw
            The raw data to apply the LCMV beamformer on.

        Returns
        -------
        stc : SourceEstimate
            The source estimate obtained by applying the LCMV beamformer.
        """
        self._check_if_prepared()
        assert self.apply_fun is not None
        assert self.filters is not None
        return self.apply_fun(data, self.filters)

    def fit_transform(  # type: ignore[override]
        self,
        data: mne.io.BaseRaw,
        data_cov: mne.Covariance | None = None,
        noise_cov: mne.Covariance | None = None,
    ) -> mne.SourceEstimate:
        """
        Fit the LCMV beamformer to the provided data and then apply it.

        Parameters
        ----------
        data : Raw
            The raw data to fit and apply the LCMV beamformer on.
        data_cov : Covariance | None, optional
            The data covariance matrix. If None, it will be computed from
            the raw data.
        noise_cov : Covariance | None, optional
            The noise covariance matrix. If None, an ad-hoc covariance matrix
            will be created.

        Returns
        -------
        stc : SourceEstimate
            The source estimate obtained by applying the LCMV beamformer.
        """
        self.fit(data, data_cov=data_cov, noise_cov=noise_cov)
        return self.transform(data)

    def get_weights(self) -> np.ndarray:
        """
        Get the weight matrix corresponding to the fitted LCMV beamformer.

        Returns
        -------
        weights : array, shape (n_sources, n_sensors)
            The weight matrix of the LCMV beamformer.
        """
        self._check_if_prepared()
        assert self._weights is not None
        return self._weights

    def get_params(self) -> dict[str, T.Any]:
        """
        Get the parameters of the LCMV beamformer.

        Returns
        -------
        params : dict
            A dictionary containing the parameters of the LCMV beamformer.
        """
        return dict(
            reg=self.reg, weight_norm=self.weight_norm, cov_tstep=self.cov_tstep
        )
