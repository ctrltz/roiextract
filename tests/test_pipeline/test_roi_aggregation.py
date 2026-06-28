import mne
import numpy as np
import pytest

from unittest.mock import MagicMock

from roiextract.pipeline.inverse import Inverse
from roiextract.pipeline.roi_aggregation import (
    SVDAggregation,
    MeanAggregation,
    CentroidAggregation,
)
from roiextract.utils import get_label_mask


@pytest.mark.parametrize(
    "flip, n_labels", [(False, 1), (True, 1), (False, 2), (True, 2)]
)
def test_mean_aggregation(default_eeg_setup, flip, n_labels):
    fwd, inv_op, raw_eeg, labels = default_eeg_setup
    src = fwd["src"]
    labels_to_use = labels[:n_labels]

    inv_step = Inverse(inv_op, method="eLORETA", lambda2=1.0 / 9.0)
    stc = inv_step.fit_transform(raw_eeg)

    agg_step = MeanAggregation(flip=flip)
    label_tc = agg_step.fit_transform(stc, src, labels=labels_to_use)
    assert label_tc.shape == (n_labels, raw_eeg.times.size)
    weights = agg_step.get_weights()

    # Apply the method using the extracted weights and ensure that the result
    # matches MNE-based computation
    extracted = weights @ stc.data
    assert np.allclose(label_tc, extracted, atol=1e-9), f"Mismatch in flip={flip}"

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

    inv_step = Inverse(inv_op, method="eLORETA", lambda2=1.0 / 9.0)
    stc = inv_step.fit_transform(raw_eeg)

    agg_step = CentroidAggregation(surf="custom")
    label_tc = agg_step.fit_transform(stc, src, labels=labels_to_use)
    assert label_tc.shape == (n_labels, raw_eeg.times.size)
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
    assert np.allclose(label_tc, extracted, atol=1e-9)

    # Check the metadata
    assert agg_step.get_names() == [
        label.name for label in labels_to_use
    ], "Row names do not match label names"
    assert agg_step.get_params()["surf"] == "custom"


@pytest.mark.parametrize("n_labels", [1, 2])
def test_svd_aggregation__one_component(default_eeg_setup, n_labels):
    fwd, inv_op, raw_eeg, labels = default_eeg_setup
    src = fwd["src"]
    labels_to_use = labels[:n_labels]

    inv_step = Inverse(inv_op, method="eLORETA", lambda2=1.0 / 9.0)
    stc = inv_step.fit_transform(raw_eeg)

    agg_step = SVDAggregation(n_components=1)
    label_tc = agg_step.fit_transform(stc, src, labels=labels_to_use)
    assert label_tc.shape == (n_labels, raw_eeg.times.size)

    # MNE-Python only allows obtaining the first SVD component, check that
    # the results match up to a sign flip and scaling factor
    mne_tc = mne.extract_label_time_course(stc, labels_to_use, src, mode="pca_flip")
    for tc_mne, tc_agg in zip(mne_tc, label_tc):
        tc_mne /= np.linalg.norm(tc_mne)
        tc_agg /= np.linalg.norm(tc_agg)
        dp = np.dot(tc_mne, tc_agg)
        assert np.isclose(
            abs(dp), 1.0, atol=1e-9
        ), "Mismatch between MNE-Python and SVDAggregation results"

    # Check the metadata
    assert agg_step.get_names() == [
        label.name for label in labels_to_use
    ], "Row names do not match label names"


def test_svd_aggregation__multiple_components(default_eeg_setup):
    fwd, inv_op, raw_eeg, labels = default_eeg_setup
    src = fwd["src"]
    label_to_use = labels[0]

    inv_step = Inverse(inv_op, method="eLORETA", lambda2=1.0 / 9.0)
    stc = inv_step.fit_transform(raw_eeg)

    agg_step = SVDAggregation(n_components=3)
    label_tc = agg_step.fit_transform(stc, src, labels=label_to_use)
    assert label_tc.shape == (3, raw_eeg.times.size)

    # Check the metadata
    expected_names = [f"{label_to_use.name} (SVD{i+1})" for i in range(3)]
    assert (
        agg_step.get_names() == expected_names
    ), "Row names do not match expected SVD component names"
