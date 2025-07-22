import mne
import numpy as np

from mne.io.constants import FIFF


def prepare_leadfield(fwd):
    free_ori_error_msg = "Free orientations are currently not supported."

    if not isinstance(fwd, mne.Forward):
        if np.atleast_2d(fwd).ndim != 2:
            raise ValueError(free_ori_error_msg)

        return fwd, False

    if fwd["source_ori"] == FIFF.FIFFV_MNE_FIXED_ORI:
        return fwd["sol"]["data"], True

    raise ValueError(free_ori_error_msg)


def prepare_covariance(cov):
    return cov
