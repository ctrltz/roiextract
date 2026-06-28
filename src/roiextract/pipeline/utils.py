import mne
import numpy as np

from mne.beamformer import apply_lcmv, Beamformer
from mne.minimum_norm import apply_inverse, InverseOperator


def _prepare_identity_evoked(ch_names: list[str], ch_types: list[str]) -> mne.Evoked:
    """
    Prepare an Evoked object with an identity matrix as data.
    This is used to obtain the weight matrix from an inverse operator.
    """
    n_chans = len(ch_names)
    dummy_info = mne.create_info(
        ch_names=ch_names,
        sfreq=1,  # Sampling frequency does not matter
        ch_types=ch_types,
    )
    evoked = mne.EvokedArray(np.eye(n_chans), dummy_info, tmin=0)
    evoked.set_eeg_reference(projection=True)
    return evoked


def _get_matrix_from_prepared_inverse_operator(
    inv_op: InverseOperator, method: str, lambda2: float
) -> np.ndarray:
    """
    Get the weight matrix from a prepared inverse operator. The same idea
    as in the MNE library is used: applying the inverse operator
    to an identity matrix gives inverse matrix as the output.
    """
    info = inv_op["info"]
    ch_names = info["ch_names"]
    ch_types = [mne.channel_type(info, idx) for idx in range(len(ch_names))]
    evoked = _prepare_identity_evoked(ch_names, ch_types)
    stc = apply_inverse(evoked, inv_op, method=method, lambda2=lambda2, prepared=True)
    return stc.data


def _get_matrix_from_lcmv_filters(info: mne.Info, filters: Beamformer) -> np.ndarray:
    """
    Get the weight matrix from a list of LCMV filters.
    """
    ch_names = filters["ch_names"]
    ch_types = [mne.channel_type(info, idx) for idx in range(len(ch_names))]
    evoked = _prepare_identity_evoked(ch_names, ch_types)
    stc = apply_lcmv(evoked, filters)
    return stc.data
