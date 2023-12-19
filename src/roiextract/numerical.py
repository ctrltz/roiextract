import numpy as np

from numpy.linalg import norm
from scipy.optimize import minimize


def _ctf_ratio(w, L, mask):
    # Split the leadfield into within and outside parts
    L_in = L[:, mask]
    L_out = L[:, ~mask]

    ctf_out = np.squeeze(w @ L_out)
    ctf_in = np.squeeze(w @ L_in)

    F_out = norm(ctf_out) ** 2
    F_in = norm(ctf_in) ** 2

    # Function and its gradient for optimization
    F = F_out / F_in
    dF = 2 * w @ (L_out @ L_out.T - F * L_in @ L_in.T) / F_in
    
    return F, dF


def _ctf_homogeneity(w, L, P0, mask):
    # Use only voxels within the mask
    L_in = L[:, mask]

    # Make sure P0 is a row vector with unit length
    P0 = np.squeeze(P0)[np.newaxis, :]
    P0 = P0 / norm(P0)

    ctf_in = np.squeeze(w @ L_in)
    P_in = ctf_in ** 2
    dotprod = np.squeeze(P_in @ P0.T)
    L_P = L_in * np.tile(P0, (L.shape[0], 1))

    F = dotprod ** 2 / norm(P_in) ** 2
    dF = 4 / (norm(P_in) ** 2) * (dotprod * ctf_in @ L_P.T - F * ctf_in ** 3 @ L_in.T)

    # multiply with -1 to minimize
    return (-1) * F, (-1) * dF


def _ctf_compromise(w, L, P0, mask, alpha):
    F_hom, dF_hom = _ctf_homogeneity(w, L, P0, mask)
    F_rat, dF_rat = _ctf_ratio(w, L, mask)

    F = alpha * F_hom + (1 - alpha) * F_rat
    dF = alpha * dF_hom + (1 - alpha) * dF_rat

    return F, dF


def ctf_optimize_ratio_homogeneity(leadfield, template, mask, alpha, 
                                   x0, return_scipy=False, **kwargs):
    result = minimize(
        _ctf_compromise, x0, 
        args=(leadfield, template, mask, alpha), 
        jac=True, **kwargs
    )
    if return_scipy:
        return result
    
    return result.x