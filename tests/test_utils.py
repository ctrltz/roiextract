import numpy as np
import mne
import pytest

from mock import patch

from roiextract.utils import (
    _check_input,
    _report_props,
    get_label_mask,
    resolve_template,
)


@pytest.mark.parametrize(
    "param,value,allowed_values",
    [["param1", 2, [1, 2, 3]], ["param2", "sim", ["sim", "rat"]]],
)
def test_check_input_should_allow(param, value, allowed_values):
    _check_input(param, value, allowed_values)


@pytest.mark.parametrize(
    "param,value,allowed_values",
    [["param1", 4, [1, 2, 3]], ["param2", "hom", ["sim", "rat"]]],
)
def test_check_input_should_not_allow(param, value, allowed_values):
    with pytest.raises(ValueError):
        _check_input(param, value, allowed_values)


@pytest.mark.parametrize(
    "props,expected_result",
    [
        [dict(rat=0.812, sim=0.0626), "rat=0.81, sim=0.063"],
        [dict(rat=0.812, hom=0.2), "rat=0.81, hom=0.2"],
        [dict(rat=0.812, sim=0.0626, hom=0.2), "rat=0.81, sim=0.063, hom=0.2"],
    ],
)
def test_report_props(props, expected_result):
    assert _report_props(props) == expected_result


@pytest.mark.parametrize(
    "hemi,vertices,expected_mask",
    [
        ["lh", [3, 5], [0, 1, 1, 0, 0, 0, 0, 0, 0]],
        ["rh", [0, 4, 8], [0, 0, 0, 0, 1, 0, 1, 0, 1]],
    ],
)
def test_get_label_mask(hemi, vertices, expected_mask):
    test_src = [{"vertno": [1, 3, 5, 7]}, {"vertno": [0, 2, 4, 6, 8]}]
    test_label = mne.Label(hemi=hemi, vertices=vertices)
    mask = get_label_mask(test_label, test_src)
    assert np.array_equal(mask, np.array(expected_mask).astype(bool))


@patch("mne.label.label_sign_flip")
@pytest.mark.parametrize(
    "initial,expected",
    [
        [
            np.array([1.0, -1.0, 1.0, -1.0, 1.0]),
            np.array([1.0, -1.0, 1.0, -1.0, 1.0]),
        ],
        ["mean_flip", np.array([1.0, -1.0, 1.0])],
        ["mean", np.array([1.0, 1.0, 1.0])],
    ],
)
def test_resolve_template(mock_label_sign_flip, initial, expected):
    mock_label_sign_flip.return_value = np.array([1, -1, 1])
    test_src = [{"vertno": [1, 3, 5, 7]}, {"vertno": [0, 2, 4, 6, 8]}]
    test_label = mne.Label(hemi="lh", vertices=[1, 3, 7])
    resolved = resolve_template(initial, test_label, test_src)
    assert np.allclose(resolved, expected, rtol=1e-6)
