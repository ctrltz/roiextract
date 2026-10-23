import numpy as np
from numpy.linalg import norm

from roiextract.analytic import _ged_matrices_ratio_custom_cov
from roiextract.custom_types import CovarianceLike, FilterLike, ForwardLike, LabelLike
from roiextract.prepare import (
    prepare_filter,
    prepare_label_mask,
    prepare_leadfield,
    prepare_source_covariance,
    prepare_template,
)


def _ctf_ratio(w: np.ndarray, leadfield: np.ndarray, mask: np.ndarray) -> float:
    ctf = np.squeeze(w @ leadfield)
    ctf_in = ctf[mask]

    return (norm(ctf_in) / norm(ctf)) ** 2


def _ctf_ratio_cov(
    w: np.ndarray, leadfield: np.ndarray, mask: np.ndarray, source_cov: np.ndarray
) -> float:
    A, B = _ged_matrices_ratio_custom_cov(leadfield, mask, source_cov)
    return (w @ A @ w.T) / (w @ B @ w.T)


def _ctf_similarity(w, leadfield, mask, template, squared=False):
    # Make sure template is a row vector with unit length
    if template.size > 1:
        template = np.squeeze(template)
    template = template[np.newaxis, :]
    template = template / norm(template)

    ctf_in = w @ leadfield[:, mask]
    if squared:
        ctf_in = ctf_in**2
    dotprod = np.squeeze(ctf_in @ template.T)

    return np.abs(dotprod / norm(ctf_in))


def ctf_ratio(
    sf: FilterLike,
    fwd: ForwardLike,
    label: LabelLike,
    source_cov: CovarianceLike | None = None,
) -> float:
    """
    Calculates the CTF ratio (norm of the CTF within the ROI divided over
    the norm of the CTF for the entire brain).

    Parameters
    ----------
    sf : SpatialFilter | array
        The spatial filter to evaluate.
    fwd : Forward | array
        The forward model (MNE object) or the lead field (provided as an array).
    label : Label | array
        Definition of the ROI as an MNE object or a binary mask.
    source_cov : Covariance | array
        Pass a custom source covariance matrix to override the default assumption
        of unit variance and zero covariance.

    Returns
    -------
    ratio : float
        The CTF ratio.
    """
    w = prepare_filter(sf)
    leadfield = prepare_leadfield(fwd)
    mask = prepare_label_mask(label, fwd)
    source_cov = prepare_source_covariance(source_cov)

    if source_cov is None:
        return _ctf_ratio(w, leadfield, mask)

    return _ctf_ratio_cov(w, leadfield, mask, source_cov)


def ctf_similarity(sf, fwd, label, template, squared=False):
    """
    Calculates the (cosine) similarity of CTF with the provided template.

    Parameters
    ----------
    sf : SpatialFilter | array
        The spatial filter to evaluate.
    fwd : Forward | array
        The forward model (MNE object) or the lead field (provided as an array).
    label : Label | array
        Definition of the ROI as an MNE object or a binary mask.
    template : "mean" | "mean_flip" | array
        Template CTF weights, provided as an array or as a pre-defined string.
    squared : bool, default=False
        If squared, that the similarity is evaluated for the squared CTF, which
        corresponds to the contributions of power. The template weights need to be
        adjusted accordingly - for example, negative values won't be meaningful.

    Returns
    -------
    similarity : float
        The CTF similarity.
    """
    w = prepare_filter(sf)
    leadfield = prepare_leadfield(fwd)
    mask = prepare_label_mask(label, fwd)
    template = prepare_template(template, label, fwd)

    return _ctf_similarity(w, leadfield, mask, template, squared=squared)


def ctf_homogeneity(sf, fwd, label):
    """
    Calculates the homogeneity of CTF in terms of power contribution.

    Parameters
    ----------
    sf : SpatialFilter | array
        The spatial filter to evaluate.
    fwd : Forward | array
        The forward model (MNE object) or the lead field (provided as an array).
    label : Label | array
        Definition of the ROI as an MNE object or a binary mask.

    Returns
    -------
    homogeneity : float
        The homogeneity of the squared CTF with the template.
    """

    return ctf_similarity(sf, fwd, label, "mean", squared=True)
