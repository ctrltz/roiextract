import numpy as np

from numpy.linalg import norm

from ._prepare import prepare_inputs
from .utils import resolve_template, get_label_mask, _check_input


def ctf_ratio(w, fwd, label, source_mask=None, kind="power"):
    """
    Calculates ratio of CTF outside/within ROI
    """
    _check_input("kind", kind, ["power", "amplitude"])
    leadfield, label_mask, _, _ = prepare_inputs(fwd, label)

    # Take into account only a subset of sources if source mask is provided
    L = leadfield.copy()
    if source_mask is not None:
        label_mask = np.logical_and(label_mask, source_mask)
        L = leadfield[:, source_mask]

    # Split lead field matrix into in and out parts
    L_in = leadfield[:, label_mask]

    # Compute the CTF
    ctf_all = np.squeeze(w @ L)
    ctf_in = np.squeeze(w @ L_in)

    # Compute the CTF ratio
    ratio = norm(ctf_in) / norm(ctf_all)
    if kind == "power":
        return ratio**2

    return ratio


def ctf_similarity(w, L, w0, mask, source_mask=None):
    """
    Cosine similarity of desired and constructed CTFs within ROI
    """

    # Take into account only a subset of sources if source mask is provided
    if source_mask is not None:
        within_mask = source_mask[mask]
        mask = np.logical_and(mask, source_mask)
        assert within_mask.sum() > 0

    # Use only voxels within the mask
    L_in = L[:, mask]

    # TODO: check that the mask contains at least one source
    assert mask.sum() > 0

    # Make sure w0 is a row vector with unit length
    if source_mask is not None:
        w0 = w0[within_mask]
    if w0.size > 1:
        w0 = np.squeeze(w0)
    w0 = w0[np.newaxis, :]
    w0 = w0 / norm(w0)

    dotprod = np.squeeze(w @ L_in @ w0.T)
    ctf = np.squeeze(w @ L_in)

    # Adjusted value in the [0, 1] range
    return np.abs(dotprod / norm(ctf))


def ctf_homogeneity(w, L, P0, mask, source_mask=None):
    """
    Homogeneity of the CTF within ROI
    """

    # Take into account only a subset of sources if source mask is provided
    if source_mask is not None:
        within_mask = source_mask[mask]
        mask = np.logical_and(mask, source_mask)
        assert within_mask.sum() > 0

    # TODO: check that the mask contains at least one source
    assert mask.sum() > 0

    # Use only voxels within the mask
    L_in = L[:, mask]

    # Make sure P0 is a row vector with unit length
    if source_mask is not None:
        P0 = P0[within_mask]
    if P0.size > 1:
        P0 = np.squeeze(P0)
    P0 = P0[np.newaxis, :]
    P0 = P0 / norm(P0)

    ctf2 = (w @ L_in) ** 2
    dotprod = np.squeeze(ctf2 @ P0.T)

    # Adjusted value in the [0, 1] range
    return np.abs(dotprod / norm(ctf2))


def ctf_quantify(w, leadfield, mask, w0=None, P0=None, source_mask=None):
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
    source_mask: array_like
        Voxel mask with shape (voxels,). Ones correspond to all voxels that should be
        considered when evaluating CTF properties.

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
    result["rat"] = ctf_ratio(w, leadfield, mask, source_mask=source_mask)
    if w0 is not None:
        result["sim"] = ctf_similarity(
            w, leadfield, w0, mask, source_mask=source_mask
        )
    if P0 is not None:
        result["hom"] = ctf_homogeneity(
            w, leadfield, P0, mask, source_mask=source_mask
        )

    return result


def ctf_quantify_label(w, fwd, label, w0=None, P0=None, source_mask=None):
    # Extract data from Forward
    leadfield = fwd["sol"]["data"]
    src = fwd["src"]

    # Create a binary mask for the ROI
    mask = get_label_mask(label, src)

    # Support pre-defined options for template weights
    if w0 is not None:
        w0 = resolve_template(w0, label, src)
    if P0 is not None:
        P0 = resolve_template(P0, label, src)

    return ctf_quantify(
        w, leadfield, mask, w0=w0, P0=P0, source_mask=source_mask
    )


def rec_quantify(w, cov_matrix, inverse, template, mask, source_mask=None):
    return ctf_quantify(
        w, cov_matrix.T @ inverse, template, mask, source_mask=source_mask
    )


def rec_quantify_label(w, fwd, label, template, source_mask=None):
    # Extract data from Forward
    leadfield = fwd["sol"]["data"]
    src = fwd["src"]

    # Create a binary mask for the ROI
    mask = get_label_mask(label, src)

    # Support pre-defined options for template weights
    template = resolve_template(template, label, src)

    return ctf_quantify(w, leadfield, template, mask, source_mask=source_mask)
