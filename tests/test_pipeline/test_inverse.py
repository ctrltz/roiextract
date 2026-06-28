import numpy as np
import pytest

from roiextract.pipeline.inverse import Inverse, LCMVBeamformer


@pytest.mark.parametrize("method", ["MNE", "dSPM", "sLORETA", "eLORETA"])
def test_inverse_across_methods(default_eeg_setup, method):
    _, inv_op, raw_eeg, _ = default_eeg_setup

    inv_step = Inverse(inv_op, method=method, lambda2=1.0 / 9.0)
    weights = inv_step.fit(raw_eeg).get_weights()

    # Apply the method via MNE pathway
    stc = inv_step.transform(raw_eeg)

    # Apply the method using the extracted weights
    data = raw_eeg.get_data()
    extracted = weights @ data

    assert np.allclose(stc.data, extracted, atol=1e-9), f"Mismatch in method {method}"

    # Check the metadata
    assert inv_step.get_params()["method"] == method
    assert inv_step.get_params()["lambda2"] == 1.0 / 9.0


@pytest.mark.parametrize("lambda2", [1.0 / 9.0, 1.0 / 4.0, 1.0 / 16.0])
def test_inverse_across_lambdas(default_eeg_setup, lambda2):
    _, inv_op, raw_eeg, _ = default_eeg_setup

    inv_step = Inverse(inv_op, method="sLORETA", lambda2=lambda2)
    weights = inv_step.fit(raw_eeg).get_weights()

    # Apply the method via MNE pathway
    stc = inv_step.transform(raw_eeg)

    # Apply the method using the extracted weights
    data = raw_eeg.get_data()
    extracted = weights @ data

    assert np.allclose(
        stc.data, extracted, atol=1e-9
    ), f"Mismatch for lambda2={lambda2}"

    # Check the metadata
    assert inv_step.get_params()["method"] == "sLORETA"
    assert inv_step.get_params()["lambda2"] == lambda2


@pytest.mark.parametrize("reg", [0.01, 0.05, 0.25])
def test_lcmv_beamformer_across_reg_values(default_eeg_setup, reg):
    fwd, _, raw_eeg, _ = default_eeg_setup

    lcmv_step = LCMVBeamformer(fwd, reg=reg)
    weights = lcmv_step.fit(raw_eeg).get_weights()

    # Apply the method via MNE pathway
    stc = lcmv_step.transform(raw_eeg)

    # Apply the method using the extracted weights
    data = raw_eeg.get_data()
    extracted = weights @ data

    assert np.allclose(stc.data, extracted, atol=1e-9), f"Mismatch for reg={reg}"

    # Check the metadata
    assert lcmv_step.get_params()["reg"] == reg
