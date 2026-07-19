import logging
import numpy as np

from roiextract.pipeline import PipelineStep


class RankDeficiencyError(Exception):
    """
    Raised when the input data is rank deficient, which leads to noise in the
    orthogonalized components.
    """

    pass


class SymmetricOrthogonalization(PipelineStep):
    """
    A pipeline step that performs symmetric orthogonalization of the input time
    series, following (Colclough et al., 2015) and being based on the
    :meth:`mne_connectivity.symmetric_orth` implementation. This step is useful
    for reducing the effects of signal leakage in source-reconstructed MEG/EEG
    data.

    Parameters
    ----------
    n_iter : int, optional
        The maximum number of iterations for the orthogonalization algorithm.
    tol : float, optional
        The tolerance for convergence. The algorithm stops when the relative
        change in the error is below this threshold.
    use_previous_d : bool, optional
        Set this to `True` to match the MNE-connectivity implementation.
    """

    def __init__(
        self, n_iter: int = 50, tol: float = 1e-06, use_previous_d: bool = False
    ) -> None:
        super().__init__()

        self.n_iter = n_iter
        self.tol = tol
        self.use_previous_d = use_previous_d

        self._weights = None

    def fit(self, data, **kwargs):
        """
        Fit the symmetric orthogonalization step to the provided data.

        Parameters
        ----------
        data : array-like
            The input data to fit the step to.

        Returns
        -------
        self : SymmetricOrthogonalization
            The fitted instance of the symmetric orthogonalization step.
        """
        self._weights = _get_symmetric_orthogonalization_weights(
            data, n_iter=self.n_iter, tol=self.tol, use_previous_d=self.use_previous_d
        )
        self.prepared = True
        return self

    def transform(self, data):  # type: ignore[override]
        """
        Apply symmetric orthogonalization to the provided data.

        Parameters
        ----------
        data : array-like
            The input data to transform.

        Returns
        -------
        transformed_data : array-like
            The transformed data after applying symmetric orthogonalization.
        """
        self._check_if_prepared()
        return self._weights @ data

    def fit_transform(self, data):  # type: ignore[override]
        return super().fit_transform(data)

    def get_weights(self):
        self._check_if_prepared()
        return self._weights

    def get_names(self, names: list[str] | None) -> list[str] | None:
        """
        Get the names of the rows of the weight matrix, which correspond to the
        orthogonalized components.

        Parameters
        ----------
        names : list of str | None
            Names that correspond to the rows of the input data. These names
            are used to as the output names for the orthogonalized components.

        Returns
        -------
        names : list of str | None
            The names of the rows of the weight matrix, which correspond to the
            orthogonalized components.
        """
        self._check_if_prepared()
        return names

    def get_params(self):
        return dict(
            n_iter=self.n_iter, tol=self.tol, use_previous_d=self.use_previous_d
        )


def _check_rank_deficiency(sigma: np.ndarray, shape: tuple[int, int]) -> None:
    """
    Raise an error if the input data is rank deficient.
    """
    rank_tol = max(shape) * sigma[0] * np.finfo(sigma.dtype).eps
    rank = np.sum(sigma > rank_tol)
    if rank < shape[0]:
        raise RankDeficiencyError(
            f"Symmetric orthogonalization is not possible due to input data "
            f"being rank deficient ({rank} < {shape[0]}). You could try to "
            f"reduce the number of ROI time series to match the rank."
        )


def _get_symmetric_orthogonalization_weights(
    Z: np.ndarray, n_iter: int, tol: float, use_previous_d: bool = False
) -> np.ndarray:
    """
    Based on the reference implementation of symmetric orthogonalization from
    MNE-connectivity, this function computes the weights that transform the input
    data `Z` into orthogonalized components. The present implementation deviates
    slightly in the following ways:

    - An error is raised if the input data are rank deficient or become rank
      deficient during the orthogonalization process. Such behavior also matches
      the reference MATLAB implementation.

    - The same D matrix is used for SVD and scaling the orthonormal components.
      This should not affect the result a lot (although np.allclose check with
      low atol fails), but feels a bit more self-consistent and uses the same
      D matrix in all steps of final weight calculation.

    MNE-connectivity implementation:
    https://mne.tools/mne-connectivity/stable/generated/mne_connectivity.symmetric_orth.html
    MATLAB implementation: https://github.com/OHBA-analysis/MEG-ROI-nets
    """
    # Following MNE-connectivity, the format of Z is transposed compared to the
    # notation of the original paper
    n_series, n_samples = Z.shape

    # "starting with D^(1) = I_n"
    d = np.ones(n_series)
    d_prev = None

    last_err = np.inf
    power = np.linalg.norm(Z, "fro") ** 2
    power = power or 1.0
    for ii in range(n_iter):
        # Maintain D matrix from the previous iteration to match the
        # reference implementation in tests
        d_prev = d.copy()

        # Z is transposed to match the notation of the paper
        U, sigma, Vh = np.linalg.svd(Z.T * d[np.newaxis, :], full_matrices=False)
        _check_rank_deficiency(sigma, (n_series, n_samples))

        O_ = Vh.T @ U.T
        d = np.einsum("ij,ij->i", Z, O_)

        err = np.linalg.norm(Z - O_ * d[:, np.newaxis], "fro") ** 2 / power
        delta = 0 if err == 0 else (last_err - err) / err
        logging.debug(f"    {ii:2d}: ε={delta:0.2e} ({err})")
        if err == 0 or delta < tol:
            logging.info(f"Convergence reached on iteration {ii}")
            break
        last_err = err
    else:
        logging.warning("Symmetric orth did not converge")

    if use_previous_d:
        # sigma and Vh are calculated with D_prev and can be re-used
        D_prev = np.diag(d_prev)
        D = np.diag(d)
        return D @ Vh.T @ np.diag(1.0 / sigma) @ Vh @ D_prev

    # Re-run the whole process with the final D matrix
    _, sigma, Vh = np.linalg.svd(Z.T * d[np.newaxis, :], full_matrices=False)
    D = np.diag(d)
    return D @ Vh.T @ np.diag(1.0 / sigma) @ Vh @ D
