import numpy as np
from numpy.linalg import norm
from scipy.linalg import eig


def _ged_matrices_ratio(leadfield, mask):
    L_in = leadfield[:, mask]

    A = L_in @ L_in.T
    B = leadfield @ leadfield.T

    return A, B


def _ged_matrices_ratio_custom_cov(leadfield, mask, source_cov):
    L_in = leadfield[:, mask]
    source_cov_in = source_cov[np.ix_(mask, mask)]

    # TODO: the calculations become significantly slower with a custom
    # matrix, we could try speedups for corner cases (e.g., fully diagonal)
    A = L_in @ source_cov_in @ L_in.T
    B = leadfield @ source_cov @ leadfield.T

    return A, B


def _ctf_optimize_ratio(leadfield, mask, reg, source_cov=None):
    # Get the GED matrices and apply regularization
    if source_cov is None:
        A, B = _ged_matrices_ratio(leadfield, mask)
    else:
        A, B = _ged_matrices_ratio_custom_cov(leadfield, mask, source_cov)
    A_reg = A + reg * np.trace(A) * np.eye(*A.shape) / A.shape[0]
    B_reg = B + reg * np.trace(B) * np.eye(*B.shape) / B.shape[0]

    # Get the eigenvector that corresponds to the smallest eigenvalue
    [eigvals, eigvecs] = eig(A_reg, B_reg)
    w = eigvecs[:, eigvals.argmax()]

    return w


def _ctf_optimize_ratio_similarity(leadfield, mask, template, priority, reg=0.001):
    """
    Optimize a linear combination of similarity with a template CTF and
    within/outside CTF ratio

    Parameters
    ----------
    leadfield: array_like
        Lead field matrix for dipoles with fixed orientation with shape
        (channels, voxels).
    template: array_like
        Template CTF pattern for voxels with shape (voxels,). Only values
        within the ROI are used.
    mask: array_like
        Voxel mask of the region of interest with shape (voxels,). Contains
        ones and zeros for voxels within and outside ROI, respectively.
    priority: float
        If 0, only ratio is optimized. If 1, only similarity. Values in between
        allow tweaking the balance between optimization for similarity or ratio.
    regA, regB: float
        Regularization parameters to ensure that it is possible to calculate the
        inverse matrices.

    Returns
    -------
    w: array
        Spatial filter produced by the optimization.

    Raises
    ------
    ValueError
        If the provided priority value is out of [0, 1] range.
    """
    if priority > 1.0 or priority < 0.0:
        raise ValueError("Value of priority should lie in [0, 1] range")

    # Make sure w0 is a row vector with unit length
    template = np.atleast_1d(np.squeeze(template))
    if template.ndim > 1:
        raise ValueError(
            f"Template weights should be a vector, got "
            f"{template.ndim} dimensions instead"
        )
    template = template / norm(template)
    template = template[np.newaxis, :]

    # Split lead field matrix into in and out parts
    L_in = leadfield[:, mask]
    L_out = leadfield[:, ~mask]

    # Solve the generalized eigenvalue problem
    A_rat = L_out @ L_out.T
    A_sim = L_in @ (template.T @ template) @ L_in.T
    A = (1 - priority) * A_rat - priority * A_sim
    A_reg = A + reg * np.trace(A) * np.eye(*A.shape) / A.shape[0]
    B = L_in @ L_in.T
    B_reg = B + reg * np.trace(B) * np.eye(*B.shape) / B.shape[0]
    [eigvals, eigvecs] = eig(A_reg, B_reg)

    # Get the eigenvector that corresponds to the smallest eigenvalue
    w = eigvecs[:, eigvals.argmin()]
    if np.dot(w @ L_in, template.T) < 0:
        w *= -1

    return w
