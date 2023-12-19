import numpy as np
import pytest

from roiextract.optimize import suggest_alpha


@pytest.mark.parametrize(
    "criteria,threshold,tol",
    [["rat", 0.95, 0.1], ["sim", 0.9, 0.01], ["hom", 0.8, 0.001]],
)
def test_suggest_alpha(criteria, threshold, tol):
    # sqrt(1 - w ** 2) should more or less fine describe the shape
    # of the ratio-similarity and ratio-homogeneity curves
    alpha = suggest_alpha(
        lambda alpha: alpha,
        lambda w: dict(
            rat=np.sqrt(1 - w**2),
            sim=np.sqrt(1 - (w - 1) ** 2),
            hom=np.sqrt(1 - (w - 1) ** 2),
        ),
        criteria,
        threshold,
        tol,
    )

    # alpha = sqrt(1 - threshold^2) is the expected result
    # in case of similarity and homogeneity, the shape is reversed -> 1 - alpha
    expected_alpha = np.sqrt(1 - threshold**2)
    if criteria != "rat":
        expected_alpha = 1 - expected_alpha
    assert np.abs(alpha - expected_alpha) < tol
