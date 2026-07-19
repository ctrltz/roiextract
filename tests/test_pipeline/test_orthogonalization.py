import numpy as np

from mne_connectivity import symmetric_orth

from roiextract.pipeline import Inverse, MeanAggregation
from roiextract.pipeline.orthogonalization import (
    _get_symmetric_orthogonalization_weights,
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
