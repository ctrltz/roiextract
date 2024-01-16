import numpy as np

from functools import partial

from .analytic import ctf_optimize_ratio_similarity
from .filter import SpatialFilter
from .numerical import ctf_optimize_ratio_homogeneity
from .quantify import ctf_quantify
from .utils import (
    get_label_mask,
    resolve_template,
    _check_input,
    _report_props,
    logger,
)


INITIAL_ALPHAS = {"rat": 0, "sim": 0.999, "hom": 0.999}
IS_DECREASING = {"rat": True, "sim": False, "hom": False}


def suggest_alpha(opt_func, quant_func, criteria, threshold, tol=0.001):
    props = quant_func(w=opt_func(alpha=INITIAL_ALPHAS[criteria]))
    crit_thresh = threshold * props[criteria]
    logger.info(
        f"Suggesting alpha to obtain at least {threshold:.2g} of max {criteria}"
    )
    logger.info(f"Properties (alpha=0): {_report_props(props)}")
    logger.info(f"Criteria threshold: {crit_thresh:.3g}")

    left, right = 0, 1
    while right - left > tol:
        mid = (left + right) / 2
        props = quant_func(w=opt_func(alpha=mid))
        logger.info(f"Properties (alpha={mid:.3g}): {_report_props(props)}")
        if IS_DECREASING[criteria]:
            if props[criteria] > crit_thresh:
                left = mid
            else:
                right = mid
        else:
            if props[criteria] > crit_thresh:
                right = mid
            else:
                left = mid

    return (left + right) / 2


def ctf_optimize(
    leadfield,
    template,
    mask,
    alpha,
    mode="similarity",
    criteria="rat",
    threshold=None,
    tol=0.001,
    reg=0.000001,
    quantify=False,
    name="",
):
    """
    Derive a spatial filter that optimizes properties of the CTF for the extracted ROI time series

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
    mode: str
        Optimize a combination of ratio and similarity (mode="similarity") or ratio and homogeneity (mode="homogeneity").
    reg: float
        Regularization parameter to ensure that it is possible to calculate the inverse matrices.
    quantify: bool
        Whether to calculate CTF properties for the optimized spatial filter.

    Returns
    -------
    w: array
        Spatial filter produced by the optimization.
    alpha: float
        The value of alpha that was used during the optimization (either the provided one
        or the one obtained through binary search if alpha='auto')

    Raises
    ------
    ValueError
        If provided alpha value is out of [0, 1] range.
    """
    _check_input("mode", mode, ["similarity", "homogeneity"])
    if alpha == "auto" and threshold is None:
        raise ValueError("Threshold should be set if alpha='auto' is used")

    # Prepare the optimization and quantification functions
    if mode == "similarity":
        opt_func = partial(
            ctf_optimize_ratio_similarity,
            leadfield=leadfield,
            template=template,
            mask=mask,
            reg=reg,
        )
        quant_func = partial(
            ctf_quantify, leadfield=leadfield, mask=mask, w0=template
        )
    else:
        x0 = np.ones((leadfield.shape[0],))
        opt_func = partial(
            ctf_optimize_ratio_homogeneity,
            leadfield=leadfield,
            template=template,
            mask=mask,
            x0=x0,
        )
        quant_func = partial(
            ctf_quantify, leadfield=leadfield, mask=mask, P0=template
        )

    # Suggest alpha if needed
    if alpha == "auto":
        alpha = suggest_alpha(
            opt_func, quant_func, criteria, threshold, tol=tol
        )
        logger.info(
            f"alpha={alpha:.2g} was selected using the {threshold:2g} threshold"
        )

    # Optimize the filter, normalize and quantify its properties if needed
    w_opt = opt_func(alpha=alpha)
    w_opt = w_opt / np.abs(w_opt).max()
    sf = SpatialFilter(w=w_opt, alpha=alpha, name=name)
    if quantify:
        props = quant_func(w=w_opt)
        return sf, props

    return sf


def ctf_optimize_label(
    fwd,
    label,
    template,
    alpha,
    mode="similarity",
    criteria="rat",
    threshold=None,
    tol=0.001,
    reg=0.00001,
    quantify=False,
    name=None,
):
    # Extract data from Forward and Label
    leadfield = fwd["sol"]["data"]
    src = fwd["src"]
    if name is None:
        name = label.name

    # Create a binary mask for the ROI
    mask = get_label_mask(label, src)

    # Support pre-defined options for template weights
    template = resolve_template(template, label, src)

    # Optimize the filter and quantify its properties if needed
    return ctf_optimize(
        leadfield,
        template,
        mask,
        alpha,
        mode=mode,
        criteria=criteria,
        threshold=threshold,
        tol=tol,
        reg=reg,
        quantify=quantify,
        name=name,
    )


def rec_optimize(
    cov_matrix,
    inverse,
    template,
    mask,
    alpha,
    mode="similarity",
    criteria="rat",
    threshold=None,
    tol=0.001,
    reg=0.00001,
    quantify=False,
    name="",
):
    return ctf_optimize(
        cov_matrix.T @ inverse,
        template,
        mask,
        alpha,
        mode=mode,
        criteria=criteria,
        threshold=threshold,
        tol=tol,
        reg=reg,
        quantify=quantify,
    )


def rec_optimize_label(
    cov_matrix,
    fwd,
    inv,
    label,
    template,
    alpha,
    mode="similarity",
    criteria="rat",
    threshold=None,
    tol=0.001,
    reg=0.00001,
    quantify=False,
    name=None,
    **inv_kwargs,
):
    from mne.minimum_norm.resolution_matrix import (
        _get_matrix_from_inverse_operator,
    )

    # Extract data from Forward and Label
    src = fwd["src"]
    if name is None:
        name = label.name

    # Extract data from InverseOperator
    inverse = _get_matrix_from_inverse_operator(inv, fwd, **inv_kwargs)

    # Create a binary mask for the ROI
    mask = get_label_mask(label, src)

    # Support pre-defined options for template weights
    template = resolve_template(template, label, src)

    # Optimize the filter and quantify its properties if needed
    return rec_optimize(
        cov_matrix,
        inverse,
        template,
        mask,
        alpha,
        mode=mode,
        criteria=criteria,
        threshold=threshold,
        tol=tol,
        reg=reg,
        quantify=quantify,
        name=name,
    )
