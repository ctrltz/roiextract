import numpy as np


from ._analytic import (
    _ctf_ratio_ged_matrices,
    _solve_ged,
)
from .filter import SpatialFilter
from ._prepare import prepare_inputs


# def ctf_optimize(
#     leadfield,
#     template,
#     mask,
#     lambda_,
#     mode="similarity",
#     criteria="rat",
#     threshold=None,
#     initial="auto",
#     tol=0.001,
#     reg=0.001,
#     quantify=False,
#     name="",
# ):
#     """
#     Derive a spatial filter that optimizes properties of the CTF for the extracted ROI time series

#     Parameters
#     ----------
#     leadfield: array_like
#         Lead field matrix for dipoles with fixed orientation with shape (channels, voxels).
#     template: array_like
#         Template CTF pattern for voxels with shape (voxels,). Only values within the ROI are used.
#     mask: array_like
#         Voxel mask of the region of interest with shape (voxels,). Contains ones and zeros for voxels within and
#         outside ROI, respectively.
#     lambda_: float
#         If 0, only ratio is optimized. If 1, only dot product. Values in between allow tweaking the balance between
#         optimization for dot product or ratio.
#     mode: str
#         Optimize a combination of ratio and similarity (mode="similarity") or ratio and homogeneity (mode="homogeneity").
#     reg: float
#         Regularization parameter to ensure that it is possible to calculate the inverse matrices.
#     quantify: bool
#         Whether to calculate CTF properties for the optimized spatial filter.

#     Returns
#     -------
#     sf: SpatialFilter
#         Spatial filter produced by the optimization.
#     props: dict, only returned if quantify=True
#         Dictionary that contains the estimates of CTF-based properties (ratio, similarity
#         and/or homogeneity).

#     Raises
#     ------
#     ValueError
#         If provided lambda_ value is out of [0, 1] range.
#     """
#     _check_input("mode", mode, ["similarity", "homogeneity"])
#     _check_input("initial", initial, ["auto", "ones", "reg"])
#     if lambda_ == "auto" and threshold is None:
#         raise ValueError("Threshold should be set if lambda_='auto' is used")

#     # Prepare the optimization and quantification functions
#     if mode == "similarity":
#         opt_func = partial(
#             ctf_optimize_ratio_similarity,
#             leadfield=leadfield,
#             template=template,
#             mask=mask,
#             regA=reg,
#         )
#         quant_func = partial(
#             ctf_quantify, leadfield=leadfield, mask=mask, w0=template
#         )
#     else:
#         # Setup the initial guess for numerical optimization
#         initial = ["ones", "reg"] if initial == "auto" else [initial]
#         x0s = []
#         if "ones" in initial:
#             x0_ones = np.ones((leadfield.shape[0],))
#             x0s.append(x0_ones)
#         if "reg" in initial:
#             # TODO: find some alternative to this formula (change lambda?)
#             regA = 0.1 if lambda_ == "auto" else np.exp(10 * lambda_**2 - 10)
#             x0_reg = ctf_optimize_ratio_similarity(
#                 leadfield, template, mask, lambda_=0, regA=regA, regB=reg
#             )
#             x0s.append(x0_reg)

#         opt_func = partial(
#             ctf_optimize_ratio_homogeneity,
#             leadfield=leadfield,
#             template=template,
#             mask=mask,
#             x0s=x0s,
#         )
#         quant_func = partial(
#             ctf_quantify, leadfield=leadfield, mask=mask, P0=template
#         )

#     # Suggest lambda if needed
#     if lambda_ == "auto":
#         lambda_ = suggest_lambda(
#             opt_func, quant_func, criteria, threshold, tol=tol
#         )
#         logger.info(
#             f"lambda={lambda_:.2g} was selected using the {threshold:2g} threshold"
#         )

#     # Optimize the filter, normalize and quantify its properties if needed
#     w_opt = opt_func(lambda_=lambda_)
#     w_opt = w_opt / np.abs(w_opt).max()
#     sf = SpatialFilter(
#         w=w_opt,
#         method="ctf_optimize",
#         method_params=dict(lambda_=lambda_),
#         name=name,
#     )
#     if quantify:
#         props = quant_func(w=w_opt)
#         return sf, props

#     return sf


def ctf_optimize_ratio(
    fwd, label, source_cov=None, regA=1e-6, regB=1e-6, wrap=True
):
    """
    Optimize the CTF ratio for the provided forward model and ROI.
    """

    leadfield, mask, _, source_cov = prepare_inputs(
        fwd=fwd, label=label, source_cov=source_cov
    )
    A, B = _ctf_ratio_ged_matrices(leadfield, mask, source_cov=source_cov)
    A_reg = A + regA * np.trace(A) * np.eye(*A.shape) / A.shape[0]
    B_reg = B + regB * np.trace(B) * np.eye(*B.shape) / B.shape[0]
    w = _solve_ged(A_reg, B_reg)

    if wrap:
        return SpatialFilter(w, method="ctf_optimize_ratio")

    return w


# def ctf_optimize_ratio_similarity(
#     fwd, label, template, lmbd, regA=1e-6, regB=1e-6
# ):
#     """
#     Optimize a linear combination of similarity with a template CTF and
#     within/outside CTF ratio

#     Parameters
#     ----------
#     leadfield: array_like
#         Lead field matrix for dipoles with fixed orientation with shape
#         (channels, voxels).
#     template: array_like
#         Template CTF pattern for voxels with shape (voxels,). Only values
#         within the ROI are used.
#     mask: array_like
#         Voxel mask of the region of interest with shape (voxels,). Contains
#         ones and zeros for voxels within and outside ROI, respectively.
#     lmbd: float
#         If 0, only ratio is optimized. If 1, only similarity. Values in between
#         allow tweaking the balance between optimization for similarity or ratio.
#     regA, regB: float
#         Regularization parameters to ensure that it is possible to calculate the
#         inverse matrices.

#     Returns
#     -------
#     w: array
#         Spatial filter produced by the optimization.

#     Raises
#     ------
#     ValueError
#         If the provided lambda_ value is out of [0, 1] range.
#     """
#     if lmbd > 1.0 or lmbd < 0.0:
#         raise ValueError("Value of lmbd should lie in [0, 1] range")

#     leadfield, label_mask, source_cov = prepare_inputs(
#         fwd, label, template, source_cov
#     )

#     # Combine the GED matrices for ratio and similarity
#     A_rat, B = _ctf_ratio_ged_matrices(leadfield, label_mask, source_cov)
#     A_sim, _ = _ctf_similarity_ged_matrices(
#         leadfield, label_mask, template, source_cov
#     )
#     A = (1 - lmbd) * A_rat - lmbd * A_sim

#     A_reg = A + regA * np.trace(A) * np.eye(*A.shape) / A.shape[0]
#     B_reg = B + regB * np.trace(B) * np.eye(*B.shape) / B.shape[0]

#     # Solve the generalized eigenvalue problem
#     w = _solve_ged(A_reg, B_reg)

#     return w


# def ctf_optimize_ratio_homogeneity(
#     fwd, label, template, lmbd, initial, **kwargs
# ):
#     result = _minimize_with_several_guesses(
#         _ctf_compromise,
#         x0s,
#         args=(leadfield, template, mask, lambda_),
#         jac=True,
#         **kwargs,
#     )
#     if return_scipy:
#         return result

#     return result.x
