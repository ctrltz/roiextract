import numpy as np

from .analytic import ctf_optimize_ratio_similarity
from .numerical import ctf_optimize_ratio_homogeneity
from .quantify import ctf_quantify, rec_quantify
from .utils import get_label_mask, resolve_template, _check_input


def ctf_optimize(leadfield, template, mask, alpha, 
                 mode='similarity', reg=0.000001, quantify=False):
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

    Raises
    ------
    ValueError
        If provided alpha value is out of [0, 1] range.
    """
    _check_input("mode", mode, ["similarity", "homogeneity"])
    if mode == "similarity":
        func = ctf_optimize_ratio_similarity
        quantify_kwargs = dict(w0=template)
    else:
        func = ctf_optimize_ratio_homogeneity
        quantify_kwargs = dict(P0=template)

    # TODO: find optimal alpha

    # Optimize the filter and quantify its properties if needed
    w_opt = func(leadfield, template, mask, alpha, reg)
    if quantify:
        props = ctf_quantify(w_opt, leadfield, mask, **quantify_kwargs)
        return w_opt, props
    
    return w_opt


def ctf_optimize_label(fwd, label, template, alpha, 
                       mode="similarity", reg=0.00001, quantify=False):
    # Extract data from Forward
    leadfield = fwd['sol']['data']
    src = fwd['src']

    # Create a binary mask for the ROI
    mask = get_label_mask(label, src)

    # Support pre-defined options for template weights
    template = resolve_template(template, label, src)

    # Optimize the filter and quantify its properties if needed
    return ctf_optimize(leadfield, template, mask, alpha, 
                        mode=mode, reg=reg, quantify=quantify)


def rec_optimize(cov_matrix, inverse, template, mask, alpha, 
                 mode="similarity", reg=0.00001, quantify=False):
    return ctf_optimize(cov_matrix.T @ inverse, template, mask, alpha, 
                        mode=mode, reg=reg, quantify=quantify)


def rec_optimize_label(cov_matrix, fwd, inv, label, template, alpha, 
                       mode="similarity", reg=0.00001, quantify=False, **inv_kwargs):
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
    return rec_optimize(cov_matrix, inverse, template, mask, alpha, 
                        mode=mode, reg=reg, quantify=quantify)
