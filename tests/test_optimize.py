import numpy as np
import pytest

from roiextract.optimize import suggest_lambda


@pytest.mark.parametrize(
    "criteria,threshold,tol",
    [["rat", 0.95, 0.1], ["sim", 0.9, 0.01], ["hom", 0.8, 0.001]],
)
def test_suggest_lambda(criteria, threshold, tol):
    # sqrt(1 - w ** 2) should more or less fine describe the shape
    # of the ratio-similarity and ratio-homogeneity curves
    lambda_ = suggest_lambda(
        lambda lambda_: lambda_,
        lambda w: dict(
            rat=np.sqrt(1 - w**2),
            sim=np.sqrt(1 - (w - 1) ** 2),
            hom=np.sqrt(1 - (w - 1) ** 2),
        ),
        criteria,
        threshold,
        tol,
    )

    # lambda = sqrt(1 - threshold^2) is the expected result
    # in case of similarity and homogeneity, the shape is reversed -> 1 - lambda
    expected_lambda = np.sqrt(1 - threshold**2)
    if criteria != "rat":
        expected_lambda = 1 - expected_lambda
    assert np.abs(lambda_ - expected_lambda) < tol
