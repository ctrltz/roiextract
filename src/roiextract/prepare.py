import mne
import numpy as np

from mne._fiff.constants import FIFF
from mne.label import label_sign_flip

from roiextract.filter import SpatialFilter
from roiextract.utils import get_label_mask, _check_input


def prepare_filter(sf):
    if not isinstance(sf, SpatialFilter):
        return sf

    return sf.w


def prepare_leadfield(fwd: mne.Forward | np.ndarray) -> np.ndarray:
    """
    Extract the lead field matrix from the provided forward model.

    Parameters
    ----------
    fwd : mne.Forward | np.ndarray
        The forward model or lead field matrix. If a forward model is provided,
        the lead field matrix will be extracted from it. Otherwise, the provided
        lead field matrix will be returned as is.

    Returns
    -------
    leadfield : np.ndarray
        The lead field matrix.
    """
    if not isinstance(fwd, mne.Forward):
        return fwd

    if fwd["source_ori"] != FIFF.FIFFV_MNE_FIXED_ORI:
        raise ValueError("Only fixed source orientations are currently supported.")

    return fwd["sol"]["data"]


def prepare_label_mask(label, fwd=None):
    if not isinstance(label, mne.Label | mne.BiHemiLabel):
        return label

    assert fwd is not None and isinstance(fwd, mne.Forward)
    return get_label_mask(label, fwd["src"])


def prepare_template(template, label=None, fwd=None):
    if isinstance(template, str):
        _check_input("template", template, ["mean_flip", "mean"])
        assert fwd is not None and isinstance(fwd, mne.Forward)
        assert label is not None and isinstance(label, mne.Label | mne.BiHemiLabel)
        signflip = label_sign_flip(label, fwd["src"])[np.newaxis, :]

        if template == "mean_flip":
            return signflip
        if template == "mean":
            return np.ones((1, signflip.size))

    return template
