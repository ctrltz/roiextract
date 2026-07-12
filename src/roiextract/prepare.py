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
    """
    Prepare a binary mask that indicates which sources in the forward model
    correspond to the given label.

    Parameters
    ----------
    label : mne.Label | mne.BiHemiLabel | np.ndarray
        The label for which to create the mask. If a numpy array is provided,
        it is assumed to contain such a binary mask and will be returned as is.
    fwd : mne.Forward, optional
        The forward model from which to extract the source space. This is required
        if an :class:`mne.Label` or :class:`mne.BiHemiLabel` is provided as the
        first argument.

    Returns
    -------
    mask : np.ndarray
        A binary mask indicating which sources in the forward model correspond to
        the given label.
    """
    if not isinstance(label, mne.Label | mne.BiHemiLabel):
        return label

    if fwd is None or not isinstance(fwd, mne.Forward):
        raise ValueError(
            "An `mne.Forward` object must be provided when using an `mne.Label` "
            "or `mne.BiHemiLabel` to define the ROI."
        )

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
