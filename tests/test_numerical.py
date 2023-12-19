import numpy as np

from mock import patch
from scipy.optimize import OptimizeResult

from roiextract.numerical import (
    _ctf_ratio, _ctf_homogeneity, _ctf_compromise, 
    ctf_optimize_ratio_homogeneity
)


def test_ctf_ratio_minimal():
    """
    Tests the most simple case with 2 channels, 2 sources and identity leadfield.
    If w = [w1 w2], then F = w2 ** 2 / w1 ** 2, 
    dF/dw1 = -2 w2 ** 2 / w1 ** 3, dF/dw2 = 2 w2 / w1 ** 2
    """
    L = np.eye(2, 2)
    mask = np.array([True, False])
    for _ in range(10):
        w = np.random.randn(2)
        expected_F = w[1] ** 2 / w[0] ** 2
        expected_dF = np.array([
            -2 * w[1] ** 2 / w[0] ** 3,
            2 * w[1] / w[0] ** 2
        ])

        F, dF = _ctf_ratio(w, L, mask)
        assert np.allclose(F, expected_F, rtol=1e-6)
        assert np.allclose(dF, expected_dF, rtol=1e-6)


def test_ctf_homogeneity_minimal():
    """
    Tests the most simple case with 2 channels, 2 sources and identity leadfield.
    If w = [w1 w2], then F = (-1) * (0.5 + w1 ** 2 * w2 ** 2 / (w1 ** 4 + w2 ** 4))
    dF/dw1 = 2 w1 w2 ** 2 * (w1 ** 4 - w2 ** 4) / (w1 ** 4 + w2 ** 4) ** 2
    dF/dw2 = 2 w2 w1 ** 2 * (w2 ** 4 - w1 ** 4) / (w1 ** 4 + w2 ** 4) ** 2
    """
    L = np.eye(2, 2)
    mask = np.array([True, True])
    P0 = np.array([1, 1])
    for _ in range(10):
        w = np.random.randn(2)
        denom = w[0] ** 4 + w[1] ** 4
        expected_F = (-1) * (0.5 + (w[0] ** 2) * (w[1] ** 2) / denom)
        expected_dF = np.array([
            2 * w[0] * (w[1] ** 2) * (w[0] ** 4 - w[1] ** 4) / denom ** 2,
            2 * w[1] * (w[0] ** 2) * (w[1] ** 4 - w[0] ** 4) / denom ** 2
        ])

        F, dF = _ctf_homogeneity(w, L, P0, mask)
        assert np.allclose(F, expected_F, rtol=1e-6)
        assert np.allclose(dF, expected_dF, rtol=1e-6)


@patch('roiextract.numerical._ctf_homogeneity')
@patch('roiextract.numerical._ctf_ratio')
def test_ctf_compromise_minimal(ratio_fn, homogeneity_fn):
    """
    Tests the linear combination with patched ratio and homogeneity.
    F_rat = 0.5, F_hom = 1 -> F = (1 - alpha) * 0.5 + alpha * 1 = 0.5 * (1 + alpha)
    dF_rat = [1 0], F_hom = [0 1] -> dF = [(1 - alpha) alpha]
    """
    ratio_fn.return_value = (0.5, np.array([1, 0]))
    homogeneity_fn.return_value = (1, np.array([0, 1]))

    for alpha in np.linspace(0, 1, num=11):
        expected_F = 0.5 * (1 + alpha)
        expected_dF = np.array([1 - alpha, alpha])

        # all inputs apart from alpha do not matter since the functions are patched
        F, dF = _ctf_compromise(None, None, None, None, alpha)
        assert np.allclose(F, expected_F, rtol=1e-6)
        assert np.allclose(dF, expected_dF, rtol=1e-6)


def test_ctf_optimize_ratio_homogeneity():
    """
    Tests only the return_scipy flag
    """
    leadfield = np.array([
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ])
    mask = np.array([True, True, False, False])
    template = np.array([1, 1])
    x0 = np.array([1, 1])

    result = ctf_optimize_ratio_homogeneity(
        leadfield, template, mask, 
        alpha=0, x0=x0, return_scipy=False
    )
    assert isinstance(result, np.ndarray)

    result = ctf_optimize_ratio_homogeneity(
        leadfield, template, mask, 
        alpha=0, x0=x0, return_scipy=True
    )
    assert isinstance(result, OptimizeResult)
