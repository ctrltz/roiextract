import numpy as np
import mne
import pytest

from mock import patch

from roiextract.filter import SpatialFilter, apply_batch, apply_batch_raw


def create_dummy_info(n_chans):
    return mne.create_info(
        [f"Ch{i+1}" for i in range(n_chans)], sfreq=1, ch_types="eeg"
    )


def test_spatialfilter_repr():
    n_chans = 10
    lambda_ = 0.5
    name = "mylabel"

    # Test __repr__ with no name set
    sf = SpatialFilter(w=np.zeros((n_chans,)), lambda_=lambda_)
    expected_repr = "<SpatialFilter | lambda=0.5 | 10 channels>"
    assert repr(sf) == expected_repr, "repr without name"

    # Test __repr__ with name
    sf.name = name
    expected_repr = "<SpatialFilter | mylabel | lambda=0.5 | 10 channels>"
    assert repr(sf) == expected_repr, "repr with name"


def test_spatialfilter_apply():
    n_chans = 5
    w = np.arange(n_chans) + 1
    data = np.eye(n_chans)

    # test apply with data matrix
    sf = SpatialFilter(w=w, lambda_=0)
    assert np.array_equal(sf.apply(data), w), "apply"

    # test apply with mne object containing the same data matrix
    info = create_dummy_info(n_chans)
    raw = mne.io.RawArray(data, info)
    assert np.array_equal(sf.apply_raw(raw), w), "apply_raw"


@pytest.mark.parametrize(
    "mode,normalize,expected",
    [
        ["amplitude", None, [-3, 4]],
        ["power", None, [9, 16]],
        ["amplitude", "norm", [-0.6, 0.8]],
        ["amplitude", "max", [-0.75, 1]],
        ["power", None, [9, 16]],
        ["power", "sum", [0.36, 0.64]],
    ],
)
def test_spatialfilter_get_ctf(mode, normalize, expected):
    # ctf = [-3, 4] or [9, 16] for power and amplitude, respectively
    w = np.array([-1.0, 2.0])
    L = np.array([[3.0, 0.0], [0.0, 2.0]])
    sf = SpatialFilter(w=w, lambda_=0)
    ctf = sf.get_ctf(L, mode=mode, normalize=normalize)
    assert np.allclose(ctf, expected, rtol=1e-6)


@patch("mne.viz.plot_topomap")
def test_spatialfilter_plot(plot_topomap_fn):
    n_chans = 5
    sf = SpatialFilter(w=np.zeros((n_chans,)), lambda_=0)
    info = create_dummy_info(n_chans)

    # Check that mne.viz.plot_topomap was called
    sf.plot(info)
    plot_topomap_fn.assert_called_once()

    # Check that it is possible to provide custom kwargs
    plot_topomap_fn.reset_mock()
    sf.plot(info, sphere="eeglab")
    plot_topomap_fn.assert_called_once()
    assert (
        plot_topomap_fn.call_args.kwargs.get("sphere") == "eeglab"
    ), "provide arg"


def test_spatialfilter_plot_bad_info():
    n_chans = 5
    sf = SpatialFilter(w=np.zeros((n_chans,)), lambda_=0)
    info = create_dummy_info(n_chans - 1)  # does not match the filter

    with pytest.raises(ValueError):
        sf.plot(info)


def test_apply_batch():
    n_chans = 5
    n_filters = 3
    w = np.arange(n_chans) + 1
    f = np.arange(n_filters) + 1
    data = np.eye(n_chans)
    expected = f[:, np.newaxis] @ w[np.newaxis, :]
    filters = [
        SpatialFilter(w=w * i, lambda_=0) for i in range(1, n_filters + 1)
    ]

    # test batch apply to the data matrix
    assert np.array_equal(apply_batch(data, filters), expected), "apply_batch"

    # test batch apply to the mne object with data matrix
    info = create_dummy_info(n_chans)
    raw = mne.io.RawArray(data, info)
    assert np.array_equal(
        apply_batch_raw(raw, filters), expected
    ), "apply_batch_raw"
