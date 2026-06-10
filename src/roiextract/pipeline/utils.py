import mne
import numpy as np

from mne.minimum_norm import apply_inverse, InverseOperator


def _get_matrix_from_prepared_inverse_operator(
    inv_op: InverseOperator, method: str, lambda2: float
) -> np.ndarray:
    """
    Get the weight matrix from a prepared inverse operator.
    """

    # Create a dummy mne.Info object with the same channels as in
    # the inverse operator
    info = inv_op["info"]
    n_chans = len(info["ch_names"])
    ch_types = [mne.channel_type(info, idx) for idx in range(n_chans)]
    dummy_info = mne.create_info(
        info["ch_names"],
        sfreq=n_chans,  # Sampling frequency does not matter
        ch_types=ch_types,
    )

    # Use the same trick as in the MNE library - applying the inverse operator
    # to an identity matrix gives inverse matrix as the output
    evoked = mne.EvokedArray(np.eye(n_chans), dummy_info, tmin=0)
    evoked.set_eeg_reference(projection=True)

    stc = apply_inverse(evoked, inv_op, method=method, lambda2=lambda2, prepared=True)
    return stc.data


def _get_matrix_from_lcmv_filters(filters: list[dict]) -> np.ndarray:
    """
    Get the weight matrix from a list of LCMV filters.
    """
    return np.array([filt["weights"] for filt in filters])
