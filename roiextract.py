import numpy as np

from numpy.linalg import norm
from scipy.linalg import eig
from scipy.optimize import minimize


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


def resolve_template(template, label, src):
    if isinstance(template, str):
        from mne.label import label_sign_flip

        signflip = label_sign_flip(label, src)[np.newaxis, :]

        if template == 'mean_flip':
            return signflip
        elif template == 'mean':
            return np.ones((1, signflip.size))
        elif template == 'svd_leadfield':
            raise NotImplementedError('svd_leadfield')
        else:
            raise ValueError(f'Bad option for template weights: {template}')

    return template


def ctf_ratio(w, L, mask):
    """
    Calculates ratio of CTF outside/within ROI and its gradient w.r.t. w

    :param w: spatial filter applied to extract ROI time series (1 x channels)
    :param L: lead field matrix (channels x voxels [x xyz])
    :param mask: binary mask of voxels belonging to the ROI
    :return:
    """

    # Split lead field matrix into in and out parts
    L_in = L[:, mask]

    ctf = np.squeeze(w @ L)
    ctf_in = np.squeeze(w @ L_in)

    # Both CTFs should be vectors
    assert ctf.ndim == 1
    assert ctf_in.ndim == 1

    # Adjusted value in the [0, 1] range
    return norm(ctf_in) / norm(ctf)


def ctf_dotprod_within(w, L, w0, mask):
    """
    Dot product of desired and constructed filters within ROI and its gradient w.r.t. w

    :param w:
    :param L:
    :param w0:
    :param mask:
    :return:
    """

    # Use only voxels within the mask
    L_in = L[:, mask]

    # Make sure w0 is a row vector with unit length
    w0 = np.squeeze(w0)[np.newaxis, :]
    w0 = w0 / norm(w0)

    dotprod = np.squeeze(w @ L_in @ w0.T)
    ctf = np.squeeze(w @ L_in)

    assert dotprod.size == 1
    assert ctf.ndim == 1

    # Adjusted value in the [0, 1] range
    return np.abs(dotprod / norm(ctf))


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
    dp = ctf_dotprod_within(w, L, w0, mask)
    rat = ctf_ratio(w, L, mask)

    return dp, rat


def ctf_quantify_label(w, fwd, label, template):
    # Extract data from Forward
    leadfield = fwd['sol']['data']
    src = fwd['src']

    # Create a binary mask for the ROI
    mask = get_label_mask(label, src)

    # Support pre-defined options for template weights
    template = resolve_template(template, label, src)

    return ctf_quantify(w, leadfield, template, mask)


def ctf_optimize(leadfield, template, mask, alpha, reg=0.000001):
    """
    Optimize CTF-based properties (dot product with a template or in/out ratio) for a spatial filter

    Parameters
    ----------
    leadfield: array_like
        Lead field matrix for dipoles with fixed orientation with shape (channels, voxels).
    template: array_like
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

    # Make sure w0 is a row vector with unit length
    template = np.squeeze(template)
    if template.ndim != 1:
        raise ValueError(f'Template weights should be a vector, got {template.ndim} dimensions instead')
    template = template / norm(template)
    template = template[np.newaxis, :]

    # Split lead field matrix into in and out parts
    L_in = leadfield[:, mask]
    L_out = leadfield[:, ~mask]

    # Solve the generalized eigenvalue problem
    A = (1 - alpha) * (L_out @ L_out.T) - alpha * L_in @ (template.T @ template) @ L_in.T
    A_reg = A + reg * np.trace(A) * np.eye(*A.shape) / A.shape[0]
    B = L_in @ L_in.T
    B_reg = B + reg * np.trace(B) * np.eye(*B.shape) / B.shape[0]
    [eigvals, eigvecs] = eig(A_reg, B_reg)

    # Get the eigenvector that corresponds to the smallest eigenvalue
    w = eigvecs[:, eigvals.argmin()]
    if np.dot(w @ L_in, template.T) < 0:
        w *= -1

    return w


def ctf_optimize_label(fwd, label, template, alpha, reg=0.00001, quantify=False):
    # Extract data from Forward
    leadfield = fwd['sol']['data']
    src = fwd['src']

    # Create a binary mask for the ROI
    mask = get_label_mask(label, src)

    # Support pre-defined options for template weights
    template = resolve_template(template, label, src)

    # Optimize the filter and quantify its properties if needed
    w = ctf_optimize(leadfield, template, mask, alpha, reg)
    if quantify:
        dp, rat = ctf_quantify(w, leadfield, template, mask)
        return w, dp, rat

    return w


def rec_quantify(w, cov_matrix, inverse, template, mask):
    return ctf_quantify(w, cov_matrix.T @ inverse, template, mask)


def rec_quantify_label(w, fwd, label, template):
    # Extract data from Forward
    leadfield = fwd['sol']['data']
    src = fwd['src']

    # Create a binary mask for the ROI
    mask = get_label_mask(label, src)

    # Support pre-defined options for template weights
    template = resolve_template(template, label, src)

    return ctf_quantify(w, leadfield, template, mask)


def rec_optimize(cov_matrix, inverse, template, mask, alpha, reg=0.00001):
    return ctf_optimize(cov_matrix.T @ inverse, template, mask, alpha, reg=reg)


def rec_optimize_label(cov_matrix, fwd, inv, label, template, alpha, reg=0.00001, quantify=False, **inv_kwargs):
    from mne.minimum_norm.resolution_matrix import _get_matrix_from_inverse_operator

    # Extract data from Forward
    src = fwd['src']

    # Extract data from InverseOperator
    inverse = _get_matrix_from_inverse_operator(inv, fwd, **inv_kwargs)

    # Create a binary mask for the ROI
    mask = get_label_mask(label, src)

    # Support pre-defined options for template weights
    template = resolve_template(template, label, src)

    # Optimize the filter and quantify its properties if needed
    w = rec_optimize(cov_matrix, inverse, template, mask, alpha, reg=reg)
    if quantify:
        dp, rat = rec_quantify(w, cov_matrix, inverse, template, mask)
        return w, dp, rat

    return w
