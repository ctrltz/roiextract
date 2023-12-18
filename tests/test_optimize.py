import numpy as np
import pytest

from roiextract.optimize import suggest_alpha


@pytest.mark.parametrize(
    "threshold,tol", [
        [0.95, 0.1],
        [0.9, 0.01],
        [0.8, 0.001]
    ]
)
def test_suggest_alpha(threshold, tol):
    # sqrt(1 - w ** 2) should more or less fine describe the shape
    # of the ratio-similarity and ratio-homogeneity curves
    alpha = suggest_alpha(lambda alpha: alpha,
                          lambda w: dict(rat=np.sqrt(1 - w ** 2)),
                          threshold, tol)
    
    # in this case, sqrt(1 - threshold ** 2) is the expected result
    expected_alpha = np.sqrt(1 - threshold ** 2)
    assert np.abs(alpha - expected_alpha) < tol