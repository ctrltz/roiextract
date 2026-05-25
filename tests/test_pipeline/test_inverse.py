import numpy as np
import pytest

from roiextract.pipeline import Inverse


@pytest.mark.parametrize("method", ["MNE", "dSPM", "sLORETA", "eLORETA"])
def test_inverse_across_methods(default_eeg_setup, method):
    _, inv_op, raw_eeg, _ = default_eeg_setup

    # Crop 10 seconds to speed up the test
    raw_eeg_crop = raw_eeg.copy().crop(tmax=10.0)

    inv_step = Inverse(inv_op, method=method, lambda2=1.0 / 9.0)
    weights = inv_step.fit(raw_eeg_crop).get_weights()

    # Apply the method via MNE pathway
    stc = inv_step.transform(raw_eeg_crop)

    # Apply the method using the extracted weights
    data = raw_eeg_crop.get_data()
    extracted = weights @ data

    assert np.allclose(stc.data, extracted, atol=1e-6), f"Mismatch in method {method}"


@pytest.mark.parametrize("lambda2", [1.0 / 9.0, 1.0 / 4.0, 1.0 / 16.0])
def test_inverse_across_lambdas(default_eeg_setup, lambda2):
    _, inv_op, raw_eeg, _ = default_eeg_setup

    # Crop 10 seconds to speed up the test
    raw_eeg_crop = raw_eeg.copy().crop(tmax=10.0)

    inv_step = Inverse(inv_op, method="sLORETA", lambda2=lambda2)
    weights = inv_step.fit(raw_eeg_crop).get_weights()

    # Apply the method via MNE pathway
    stc = inv_step.transform(raw_eeg_crop)

    # Apply the method using the extracted weights
    data = raw_eeg_crop.get_data()
    extracted = weights @ data

    assert np.allclose(
        stc.data, extracted, atol=1e-6
    ), f"Mismatch for lambda2={lambda2}"
