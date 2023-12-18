import numpy as np

from numpy.linalg import norm

from .utils import resolve_template, get_label_mask


def ctf_ratio(w, L, mask):
    """
    Calculates ratio of CTF outside/within ROI
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


def ctf_similarity(w, L, w0, mask):
    """
    Cosine similarity of desired and constructed CTFs within ROI
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


def ctf_homogeneity(w, L, P0, mask):
    """
    Homogeneity of the CTF within ROI
    """
    # Use only voxels within the mask
    L_in = L[:, mask]

    # Make sure P0 is a row vector with unit length
    P0 = np.squeeze(P0)[np.newaxis, :]
    P0 = P0 / norm(P0)

    ctf2 = np.squeeze(w @ L_in) ** 2
    dotprod = np.squeeze(ctf2 @ P0.T)

    assert dotprod.size == 1
    assert ctf2.ndim == 1

    # Adjusted value in the [0, 1] range
    return np.abs(dotprod / norm(ctf2))


def ctf_quantify(w, leadfield, mask, w0=None, P0=None):
    """
    Calculate similarity within ROI and in/out ratio for a given filter

    Parameters
    ----------
    w: array_like
        Spatial filter that needs to be quantified in terms of CTF.
    L: array_like
        Lead field matrix for dipoles with fixed orientation with shape (channels, voxels).
    mask: array_like
        Voxel mask of the region of interest with shape (voxels,). Contains ones and zeros for voxels within and
        outside ROI, respectively.
    w0: array_like
        Template CTF pattern for voxels within ROI with shape (voxels_roi,).
    P0: array_like
        Template power contribution for voxels within ROI with shape (voxels_roi,).

    Returns
    -------
    result: a dictionary with the estimated CTF properties
        'rat': float
            Ratio of CTF within ROI and total CTF, lies in [0, 1]. 
        'sim': float
            Similarity between the actual CTF and the template CTF pattern, lies in [0, 1]. It is only returned if w0 was provided.
        'hom': float
            Homogeneity of the power contributions of voxels within the ROI, lies in [0, 1]. It is only returned if P0 was provided. 
    
    """

    result = dict()
    result['rat'] = ctf_ratio(w, leadfield, mask)
    if w0 is not None:
        result['sim'] = ctf_similarity(w, leadfield, w0, mask)
    if P0 is not None:
        result['hom'] = ctf_homogeneity(w, leadfield, P0, mask)

    return result


def ctf_quantify_label(w, fwd, label, template):
    # Extract data from Forward
    leadfield = fwd['sol']['data']
    src = fwd['src']

    # Create a binary mask for the ROI
    mask = get_label_mask(label, src)

    # Support pre-defined options for template weights
    template = resolve_template(template, label, src)

    return ctf_quantify(w, leadfield, template, mask)


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
