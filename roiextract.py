import numpy as np

from numpy.linalg import norm
from scipy.linalg import eig
from scipy.optimize import minimize


def _ctf_ratio(w, L, mask):
    """
    Calculates ratio of CTF outside/within ROI and its gradient w.r.t. w

    :param w: spatial filter applied to extract ROI time series (1 x channels)
    :param L: lead field matrix (channels x voxels [x xyz])
    :param mask: binary mask of voxels belonging to the ROI
    :return:
    """

    # TODO: check input arguments

    # Split lead field matrix into in and out parts
    L_in = L[:, mask, :]
    L_out = L[:, ~mask, :]

    ctf = w @ L
    ctf_out = w @ L_out
    ctf_in = w @ L_in

    # Both CTFs should be vectors
    assert np.squeeze(ctf_out).ndim == 1
    assert np.squeeze(ctf_in).ndim == 1

    F_out = norm(ctf_out) ** 2
    F_in = norm(ctf_in) ** 2

    # Function and its gradient for optimization
    F = F_out / F_in
    dF = 2 * w @ (L_out @ L_out.T - F * L_in @ L_in.T) / F_in

    # Adjusted value in the [0, 1] range
    F_adj = norm(ctf_in) / norm(ctf)

    return F, dF, F_adj


def _ctf_dotprod_within(w, L, w0, mask):
    """
    Dot product of desired and constructed filters within ROI and its gradient w.r.t. w

    :param w:
    :param L:
    :param w0:
    :param mask:
    :return:
    """

    # Use only voxels within the mask
    w0_in = w0[mask]
    L_in = L[:, mask]

    # Make sure w0 is unit length
    w0_in = w0_in / norm(w0_in)

    dotprod = w @ L_in @ w0_in.T
    ctf = w @ L_in

    assert np.isscalar(dotprod)
    assert np.squeeze(ctf).ndim == 1

    # Function and its gradient for optimization
    F = (-1) * (dotprod ** 2) / (norm(ctf) ** 2)
    dF = (-2) * w @ (F * L_in @ L_in.T + L_in @ (w0_in.T @ w0_in) @ L_in.T) / (norm(ctf) ** 2)

    # Adjusted value in the [0, 1] range
    F_adj = np.abs(dotprod / norm(ctf))

    return F, dF, F_adj


def _ctf_compromise(w, L, w0, mask, alpha):
    """
    Optimize dot product within ROI and in/out ratio simultaneously

    :param w:
    :param L:
    :param w0:
    :param mask:
    :param alpha:
    :return:
    """
    F_dp, dF_dp, _ = _ctf_dotprod_within(w, L, w0, mask)
    F_rat, dF_rat, _ = _ctf_ratio(w, L, mask)

    F = alpha * F_dp + (1 - alpha) * F_rat
    dF = alpha * dF_dp + (1 - alpha) * dF_rat

    return F, dF


def ctf_quantify(w, L, w0, mask):
    """
    Calculate dot product within ROI and in/out ratio for a given filter

    Parameters
    ----------
    w: array_like
        Spatial filter that needs to be quantified in terms of CTF.
    L: array_like
        Lead field matrix for dipoles with fixed orientation with shape (channels, voxels).
    w0: array_like
        Template CTF pattern for voxels within ROI with shape (voxels_roi,).
    mask: array_like
        Voxel mask of the region of interest with shape (voxels,). Contains ones and zeros for voxels within and
        outside ROI, respectively.

    Returns
    -------
    dp: float
        Absolute value of the dot product of CTF within the ROI and template CTF pattern, lies in [0, 1].
    rat: float
        Ratio of CTF within ROI and total CTF, lies in [0, 1].
    """
    _, _, dp = _ctf_dotprod_within(w, L, w0, mask)
    _, _, rat = _ctf_ratio(w, L, mask)

    return dp, rat


def ctf_optimize_numerical(L, w0, mask, alpha, x0, **kwargs):
    """
    Optimize CTF-based properties (dot product with a template or in/out ratio) for a spatial filter

    Parameters
    ----------
    L: array_like
        Lead field matrix for dipoles with fixed orientation with shape (channels, voxels).
    w0: array_like
        Template CTF pattern for voxels within ROI with shape (voxels_roi,).
    mask: array_like
        Voxel mask of the region of interest with shape (voxels,). Contains ones and zeros for voxels within and
        outside ROI, respectively.
    alpha: float
        If 0, only ratio is optimized. If 1, only dot product. Values in between allow tweaking the balance between
        optimization for dot product or ratio.
    x0: array_like
        Initial guess for the optimization with shape (channels,)
    kwargs: dict
        Dictionary with arguments that should be forwarded to minimize.

    Returns
    -------
    res: scipy.optimize.OptimizeResult
        The optimization result represented as a OptimizeResult object. Solution is stored in the res.x field.

    Raises
    ------
    ValueError
        If provided alpha value is out of [0, 1] range.
    """
    if alpha > 1.0 or alpha < 0.0:
        raise ValueError('Value of alpha should lie in [0, 1] range')

    return minimize(_ctf_compromise, x0, args=(L, w0, mask, alpha), jac=True, **kwargs)


def ctf_optimize(L, w0, mask, alpha, reg=0.000001):
    """
    Optimize CTF-based properties (dot product with a template or in/out ratio) for a spatial filter

    Parameters
    ----------
    L: array_like
        Lead field matrix for dipoles with fixed orientation with shape (channels, voxels).
    w0: array_like
        Template CTF pattern for voxels with shape (voxels,). Only values within the ROI are used.
    mask: array_like
        Voxel mask of the region of interest with shape (voxels,). Contains ones and zeros for voxels within and
        outside ROI, respectively.
    alpha: float
        If 0, only ratio is optimized. If 1, only dot product. Values in between allow tweaking the balance between
        optimization for dot product or ratio.
    reg: float, default is 0.000001
        Regularization parameter to ensure that it is possible to calculate the inverse matrices.

    Returns
    -------
    w: array
        Spatial filter produced by the optimization.

    Raises
    ------
    ValueError
        If provided alpha value is out of [0, 1] range.
    """
    if alpha > 1.0 or alpha < 0.0:
        raise ValueError('Value of alpha should lie in [0, 1] range')

    # Split lead field matrix into in and out parts
    L_in = L[:, mask]
    L_out = L[:, ~mask]

    # Solve the generalized eigenvalue problem
    A = (1 - alpha) * (L_out @ L_out.T) - alpha * L_in @ (w0.T @ w0) @ L_in.T
    A_reg = A + reg * np.trace(A) * np.eye(*A.shape) / A.shape[0]
    B = L_in @ L_in.T
    B_reg = B + reg * np.trace(B) * np.eye(*B.shape) / B.shape[0]
    [eigvals, eigvecs] = eig(A_reg, B_reg)

    # Get the eigenvector that corresponds to the smallest eigenvalue
    w = eigvecs[:, eigvals.argmin()]
    if np.dot(w @ L_in, w0.T) < 0:
        w *= -1

    return w


def ctf_optimize_label(L, label, src, w0, alpha, reg=0.00001, quantify=False):
    from mne.label import label_sign_flip

    # Create a binary mask for the ROI
    mask = get_label_mask(label, src)

    # Support pre-defined options for w0
    if isinstance(w0, str):
        if w0 == 'mean_flip':
            w0 = label_sign_flip(label, src)[np.newaxis, :]
        elif w0 == 'svd_leadfield':
            raise NotImplementedError('svd_leadfield')
        else:
            raise ValueError(f'Bad option for template weights: {w0}')

    # Optimize the filter and quantify its properties if needed
    w = ctf_optimize(L, w0, mask, alpha, reg)
    if quantify:
        dp, rat = ctf_quantify(w, L, w0, mask)
        return w, dp, rat

    return w


def rec_optimize_label(C, W, label, src, w0, alpha, reg=0.00001, quantify=False):
    return ctf_optimize_label(C.T @ W, label, src, w0, alpha, reg=reg, quantify=quantify)


def get_label_mask(label, src):
    vertno = [s['vertno'] for s in src]
    nvert = [len(vn) for vn in vertno]
    if label.hemi == 'lh':
        this_vertices = np.intersect1d(vertno[0], label.vertices)
        vert = np.searchsorted(vertno[0], this_vertices)
    elif label.hemi == 'rh':
        this_vertices = np.intersect1d(vertno[1], label.vertices)
        vert = nvert[0] + np.searchsorted(vertno[1], this_vertices)
    else:
        raise ValueError('label %s has invalid hemi' % label.name)

    mask = np.zeros((sum(nvert),), dtype=int)
    mask[vert] = 1
    return mask > 0
