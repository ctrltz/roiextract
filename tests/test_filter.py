import numpy as np
import mne
import pytest
import warnings

from mock import patch

from roiextract.filter import SpatialFilter, apply_batch, apply_batch_raw, dot


def create_dummy_info(n_chans):
    return mne.create_info(
        [f"Ch{i+1}" for i in range(n_chans)], sfreq=1, ch_types="eeg"
    )


def test_spatialfilter_init_should_pass():
    SpatialFilter(w=np.zeros((2,)))
    SpatialFilter(w=np.zeros((2,)), ch_names=["ch1", "ch2"])


def test_spatialfilter_init_raises():
    with pytest.raises(ValueError, match="The number of channel names"):
        SpatialFilter(w=np.zeros((2,)), ch_names=["ch1"])

    with pytest.raises(ValueError, match="should be unique."):
        SpatialFilter(w=np.zeros((2,)), ch_names=["ch1", "ch1"])


def test_spatialfilter_size():
    sf = SpatialFilter(w=np.ones((10,)))
    assert sf.size == 10


def test_spatialfilter_align_no_alignment_possible():
    with warnings.catch_warnings(record=True) as w:
        sf = SpatialFilter(w=np.ones((3,)))
        result = sf._align(3, None)

    assert len(w) == 1
    assert np.array_equal(result, [0, 1, 2])


def test_spatialfilter_align_size_mismatch():
    with pytest.raises(ValueError, match="The number of channels"):
        sf = SpatialFilter(w=np.ones((3,)))
        sf._align(2, None)


def test_spatialfilter_align_mapping_fails():
    n_chans = 3
    filter_names = np.array(["ch1", "ch2", "ch3"])
    raw_names = np.array(["ch4", "ch1", "ch3"])
    sf = SpatialFilter(w=np.ones((n_chans,)), ch_names=filter_names)
    with pytest.raises(ValueError, match="data object: ch2"):
        sf._align(n_chans, raw_names)


def test_spatialfilter_align_mapping_required():
    n_chans = 3
    filter_names = np.array(["ch1", "ch2", "ch3"])
    raw_names = np.array(["ch2", "ch1", "ch3"])
    sf = SpatialFilter(w=np.ones((n_chans,)), ch_names=filter_names)
    mapping = sf._align(n_chans, raw_names)

    assert np.array_equal(filter_names[mapping], raw_names)


def test_spatialfilter_repr():
    n_chans = 10
    name = "mylabel"
    method = "mymethod"
    method_params = dict(lambda2=0.001)

    # Test __repr__ with no name set
    sf = SpatialFilter(w=np.zeros((n_chans,)))
    expected_repr = "<SpatialFilter | 10 channels>"
    assert repr(sf) == expected_repr, "repr without name and method"

    # Test __repr__ with name
    sf.name = name
    expected_repr = "<SpatialFilter | mylabel | 10 channels>"
    assert repr(sf) == expected_repr, "repr with name but not method"

    # Test __repr__ with name and method
    sf.method = method
    sf.method_params = method_params
    expected_repr = "<SpatialFilter | mylabel | mymethod (lambda2=0.001) | 10 channels>"
    assert repr(sf) == expected_repr, "repr with name and method"


def test_spatialfilter_apply():
    n_chans = 5
    w = np.atleast_2d(np.arange(n_chans) + 1)
    data = np.eye(n_chans)

    # test apply with data matrix
    sf = SpatialFilter(w=w)
    assert np.array_equal(sf.apply(data), w), "apply"

    # test apply with mne object containing the same data matrix
    info = create_dummy_info(n_chans)
    raw = mne.io.RawArray(data, info)
    assert np.array_equal(sf.apply_raw(raw), w), "apply_raw"


def test_spatialfilter_apply_with_alignment():
    n_samples = 5
    w = np.atleast_2d(np.array([1, 1, -1]))
    data = np.tile(np.array([[1], [3], [2]]), (1, n_samples))
    sf = SpatialFilter(w=w, ch_names=["ch1", "ch2", "ch3"])

    # The signal should cancel out if the channels are mapped correctly
    expected = np.atleast_2d(np.zeros((n_samples,)))
    assert np.array_equal(sf.apply(data, ch_names=["ch1", "ch3", "ch2"]), expected)


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
    sf = SpatialFilter(w=w)
    ctf = sf.get_ctf(L, mode=mode, normalize=normalize)
    assert np.allclose(ctf, expected, rtol=1e-6)


@patch("mne.viz.plot_topomap")
def test_spatialfilter_plot(plot_topomap_fn):
    n_chans = 5
    sf = SpatialFilter(w=np.zeros((n_chans,)))
    info = create_dummy_info(n_chans)

    # Check that mne.viz.plot_topomap was called
    sf.plot(info)
    plot_topomap_fn.assert_called_once()

    # Check that it is possible to provide custom kwargs
    plot_topomap_fn.reset_mock()
    sf.plot(info, sphere="eeglab")
    plot_topomap_fn.assert_called_once()
    assert plot_topomap_fn.call_args.kwargs.get("sphere") == "eeglab", "provide arg"


def test_spatialfilter_plot_bad_info():
    n_chans = 5
    sf = SpatialFilter(w=np.zeros((n_chans,)))
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
    filters = [SpatialFilter(w=w * i) for i in range(1, n_filters + 1)]

    # test batch apply to the data matrix
    assert np.array_equal(apply_batch(data, filters), expected), "apply_batch"

    # test batch apply to the mne object with data matrix
    info = create_dummy_info(n_chans)
    raw = mne.io.RawArray(data, info)
    assert np.array_equal(apply_batch_raw(raw, filters), expected), "apply_batch_raw"


def test_apply_batch_with_alignment():
    n_samples = 5
    w = np.atleast_2d(np.array([1, 1, -1]))
    data = np.tile(np.array([[1], [3], [2]]), (1, n_samples))
    sf = SpatialFilter(w=w, ch_names=["ch1", "ch2", "ch3"])

    # The signal should cancel out if the channels are mapped correctly
    expected = np.atleast_2d(np.zeros((n_samples,)))
    actual = apply_batch(data, [sf], ch_names=["ch1", "ch3", "ch2"])
    assert np.array_equal(actual, expected)


def test_dot():
    sf1 = SpatialFilter(w=np.array([1.0, 0.0]))
    sf2 = SpatialFilter(w=np.array([0.0, 1.0]))
    assert np.isclose(dot(sf1, sf2), 0.0)

    sf1 = SpatialFilter(w=np.array([1.0, 0.0]))
    sf2 = SpatialFilter(w=np.array([1.0, 0.0]))
    assert np.isclose(dot(sf1, sf2), 1.0)


def test_dot_normalize():
    sf1 = SpatialFilter(w=np.array([1.0, 2.0]))
    sf2 = SpatialFilter(w=np.array([1.0, 2.0]))
    assert np.isclose(dot(sf1, sf2), 1.0)
    assert np.isclose(dot(sf1, sf2, normalize=False), 5.0)
