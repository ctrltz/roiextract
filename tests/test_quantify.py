import numpy as np
import pytest

from roiextract.quantify import (
    ctf_ratio,
    ctf_similarity,
    ctf_homogeneity,
    ctf_quantify,
)


def create_data():
    # Minimal w and L -> ctf = [1 1 2 2]
    L = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])
    w = np.array([1, 2])
    return w, L


@pytest.mark.parametrize(
    "mask,expected_ratio",
    [
        [np.array([True, False, False, False]), np.sqrt(0.1)],
        [np.array([True, True, False, False]), np.sqrt(0.2)],
        [np.array([False, True, True, False]), np.sqrt(0.5)],
        [np.array([True, True, True, True]), 1.0],
    ],
)
def test_ctf_ratio(mask, expected_ratio):
    w, L = create_data()
    ratio = ctf_ratio(w, L, mask)
    assert np.allclose(ratio, expected_ratio, rtol=1e-6)


@pytest.mark.parametrize(
    "mask,w0,expected_similarity",
    [
        [np.array([True, True, False, False]), np.array([1, 1]), 1.0],
        [np.array([True, True, False, False]), np.array([10, 10]), 1.0],
        [np.array([True, True, False, False]), np.array([1, -1]), 0.0],
        [np.array([False, True, True, False]), np.array([1, 2]), 1.0],
    ],
)
def test_ctf_similarity(mask, w0, expected_similarity):
    w, L = create_data()
    similarity = ctf_similarity(w, L, w0, mask)
    assert np.allclose(similarity, expected_similarity, rtol=1e-6)


@pytest.mark.parametrize(
    "mask,P0,expected_homogeneity",
    [
        [np.array([True, True, False, False]), np.array([1, 1]), 1.0],
        [np.array([True, True, False, False]), np.array([10, 10]), 1.0],
        [np.array([True, True, False, False]), np.array([1, -1]), 0.0],
        [np.array([False, True, True, False]), np.array([1, 4]), 1.0],
    ],
)
def test_ctf_homogeneity(mask, P0, expected_homogeneity):
    w, L = create_data()
    homogeneity = ctf_homogeneity(w, L, P0, mask)
    assert np.allclose(homogeneity, expected_homogeneity, rtol=1e-6)


@pytest.mark.parametrize(
    "w0,P0,expected_props",
    [
        [None, None, {"rat": 1.0}],
        [np.array([1, -1, 1, -1]), None, {"rat": 1.0, "sim": 0.0}],
        [None, np.array([1, 1, 4, 4]), {"rat": 1.0, "hom": 1.0}],
        [
            np.array([1, -1, 1, -1]),
            np.array([1, 1, 4, 4]),
            {"rat": 1.0, "sim": 0.0, "hom": 1.0},
        ],
    ],
)
def test_ctf_quantify(w0, P0, expected_props):
    w, L = create_data()
    mask = np.array([True, True, True, True])
    props = ctf_quantify(w, L, mask, w0=w0, P0=P0)
    assert props.keys() == expected_props.keys()
    vals_actual = np.array([props[k] for k in expected_props])
    vals_expected = np.array([expected_props[k] for k in expected_props])
    assert np.allclose(vals_actual, vals_expected, rtol=1e-6)
