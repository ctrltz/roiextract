import numpy as np
import pytest

from mne_connectivity import symmetric_orth
from unittest.mock import patch

from roiextract.pipeline import Inverse, MeanAggregation
from roiextract.pipeline.orthogonalization import (
    _get_symmetric_orthogonalization_weights,
    _check_rank_deficiency,
    RankDeficiencyError,
    SymmetricOrthogonalization,
)


def test_get_symmetric_orthogonalization_weights(default_eeg_setup):
    fwd, inv_op, raw_eeg, labels = default_eeg_setup
    src = fwd["src"]
    labels_to_use = labels[:20]

    inv_step = Inverse(inv_op, method="eLORETA", lambda2=1.0 / 9.0)
    stc = inv_step.fit_transform(raw_eeg)

    agg_step = MeanAggregation(flip=True)
    label_tc = agg_step.fit_transform(stc, src, labels=labels_to_use)

    # Common parameters
    n_iter = 100
    tol = 1e-6

    # Use d from previous iteration to match MNE implementation
    weights = _get_symmetric_orthogonalization_weights(
        label_tc, n_iter=n_iter, tol=tol, use_previous_d=True
    )

    orth_tc = weights @ label_tc
    mne_tc = symmetric_orth(label_tc, n_iter=n_iter, tol=tol)

    # Both time courses have values on the order of 1e-9, so decreasing atol
    assert np.allclose(
        orth_tc, mne_tc, atol=1e-13
    ), "Symmetric orthogonalization does not match MNE implementation"


def test_check_rank_deficiency():
    # Create a rank-deficient matrix (2 identical rows)
    data = np.array([[1, 2, 3], [1, 2, 3], [4, 5, 6]])
    _, sigma, _ = np.linalg.svd(data, full_matrices=False)

    with pytest.raises(RankDeficiencyError):
        _check_rank_deficiency(sigma, data.shape)


@patch(
    "roiextract.pipeline.orthogonalization._get_symmetric_orthogonalization_weights",
    return_value=np.eye(3),
)
def test_symmetric_orthogonalization__metadata(mock_get_weights, default_eeg_setup):
    fwd, inv_op, raw_eeg, labels = default_eeg_setup
    src = fwd["src"]
    labels_to_use = labels[:3]

    inv_step = Inverse(inv_op, method="eLORETA", lambda2=1.0 / 9.0)
    stc = inv_step.fit_transform(raw_eeg)

    agg_step = MeanAggregation(flip=True)
    label_tc = agg_step.fit_transform(stc, src, labels=labels_to_use)

    # Common parameters
    n_iter = 100
    tol = 1e-6

    orth_step = SymmetricOrthogonalization(n_iter=n_iter, tol=tol, use_previous_d=False)
    orth_tc = orth_step.fit_transform(label_tc)
    assert orth_tc.shape == label_tc.shape, "Output shape mismatch"

    mock_get_weights.assert_called_once_with(
        label_tc, n_iter=n_iter, tol=tol, use_previous_d=False
    )
    assert np.allclose(
        orth_step.get_weights(), np.eye(3)
    ), "Weights do not match expected identity matrix"

    assert orth_step.get_params() == {
        "n_iter": n_iter,
        "tol": tol,
        "use_previous_d": False,
    }, "Metadata parameters do not match expected values"

    prev_names = ["A", "B", "C"]
    assert (
        orth_step.get_names(prev_names) == prev_names
    ), "Names do not match expected values"
