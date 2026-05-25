import mne
import numpy as np
import pytest

from unittest.mock import MagicMock

from roiextract.pipeline import Inverse, MeanAggregation, CentroidAggregation
from roiextract.utils import get_label_mask


@pytest.mark.parametrize(
    "flip, n_labels", [(False, 1), (True, 1), (False, 2), (True, 2)]
)
def test_mean_aggregation(default_eeg_setup, flip, n_labels):
    fwd, inv_op, raw_eeg, labels = default_eeg_setup
    src = fwd["src"]
    labels_to_use = labels[:n_labels]

    # Crop 10 seconds to speed up the test
    raw_eeg_crop = raw_eeg.copy().crop(tmax=10.0)

    inv_step = Inverse(inv_op, method="eLORETA", lambda2=1.0 / 9.0)
    stc = inv_step.fit_transform(raw_eeg_crop)

    agg_step = MeanAggregation(flip=flip)
    label_tc = agg_step.fit_transform(stc, src, labels=labels_to_use)
    assert label_tc.shape == (n_labels, raw_eeg_crop.times.size)
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
    assert agg_step.prepared

    for i, label in enumerate(labels_to_use):
        mask = get_label_mask(label, src)
        assert weights[i, ~mask].nnz == 0, "Non-zero weights outside the label"
        if flip:
            sign_flip = mne.label_sign_flip(label, src)
            assert np.allclose(weights[i, mask].toarray(), sign_flip / mask.sum())
        else:
            assert np.allclose(weights[i, mask].toarray(), 1 / mask.sum())


@pytest.mark.parametrize("n_labels", [1, 2])
def test_centroid_aggregation(default_eeg_setup, n_labels):
    fwd, inv_op, raw_eeg, labels = default_eeg_setup
    src = fwd["src"]
    labels_to_use = labels[:n_labels]

    # Mock the `center_of_mass` method for labels during testing
    # Pick vertno based on indices (first label - 0, second label - 10)
    mocks = []
    for idx, label in enumerate(labels_to_use):
        hemi_idx = 0 if label.hemi == "lh" else 1
        vertno = src[hemi_idx]["vertno"][10 * idx]
        label.center_of_mass = MagicMock(return_value=vertno)
        mocks.append(label.center_of_mass)

    # Crop 10 seconds to speed up the test
    raw_eeg_crop = raw_eeg.copy().crop(tmax=10.0)

    inv_step = Inverse(inv_op, method="eLORETA", lambda2=1.0 / 9.0)
    stc = inv_step.fit_transform(raw_eeg_crop)

    agg_step = CentroidAggregation(surf="custom")
    label_tc = agg_step.fit_transform(stc, src, labels=labels_to_use)
    assert label_tc.shape == (n_labels, raw_eeg_crop.times.size)
    weights = agg_step.get_weights()

    # Check the weights
    assert weights.sum() == n_labels
    assert weights[0, 0] == 1  # 0-th in lh
    if n_labels > 1:
        assert weights[1, src[0]["nuse"] + 10] == 1  # 10-th in rh

    # Check that 'surf' is forwarded to the label's `center_of_mass` method
    for mock in mocks:
        assert mock.call_args.kwargs["surf"] == "custom"

    # Apply the method using the extracted weights and ensure that the result
    # matches MNE-based computation
    extracted = weights @ stc.data
    assert np.allclose(label_tc, extracted, atol=1e-6)

    # Check the metadata
    assert agg_step.get_names() == [
        label.name for label in labels_to_use
    ], "Row names do not match label names"
    assert agg_step.get_params()["surf"] == "custom"
    assert agg_step.prepared
