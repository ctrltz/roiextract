import mne
import numpy as np
import pytest

from roiextract.pipeline import Inverse, MeanAggregation
from roiextract.utils import get_label_mask


@pytest.mark.parametrize("flip", [False, True])
def test_mean_aggregation(default_eeg_setup, flip):
    fwd, inv_op, raw_eeg, labels = default_eeg_setup
    src = fwd["src"]
    labels_to_use = labels[:2]

    # Crop 10 seconds to speed up the test
    raw_eeg_crop = raw_eeg.copy().crop(tmax=10.0)

    inv_step = Inverse(inv_op, method="eLORETA", lambda2=1.0 / 9.0)
    stc = inv_step.fit_transform(raw_eeg_crop)

    agg_step = MeanAggregation(flip=flip)
    label_tc = agg_step.fit_transform(stc, src, labels=labels_to_use)
    assert label_tc.shape == (2, raw_eeg_crop.times.size)
    weights = agg_step.get_weights()

    # Apply the method using the extracted weights and ensure that the result
    # matches MNE-based computation
    extracted = weights @ stc.data
    assert np.allclose(label_tc, extracted, atol=1e-6), f"Mismatch in flip={flip}"

    # Check the metadata
    assert ("Flip" in repr(agg_step)) == flip
    assert agg_step.get_names() == [
        label.name for label in labels_to_use
    ], "Row names do not match label names"
    assert agg_step.get_params()["flip"] == flip

    for i, label in enumerate(labels_to_use):
        mask = get_label_mask(label, src)
        assert weights[i, ~mask].nnz == 0, "Non-zero weights outside the label"
        if flip:
            sign_flip = mne.label_sign_flip(label, src)
            assert np.allclose(weights[i, mask].toarray(), sign_flip / mask.sum())
        else:
            assert np.allclose(weights[i, mask].toarray(), 1 / mask.sum())
