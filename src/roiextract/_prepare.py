import numpy as np
import mne

from numpy.linalg import norm
from mne.io.constants import FIFF

from .utils import get_label_mask, resolve_template


def prepare_leadfield(fwd):
    if isinstance(fwd, mne.Forward):
        if fwd["source_ori"] == FIFF.FIFFV_MNE_FREE_ORI:
            raise ValueError(
                "The provided forward operator is computed for free source "
                "orientations, which are currently not supported."
            )

        return fwd["sol"]["data"]

    fwd = np.atleast_2d(np.array(fwd))
    if fwd.ndim > 2:
        raise ValueError(
            "The provided leadfield matrix has more than two dimensions, "
            "but free source orientations are currently not supported."
        )

    return fwd


def prepare_label_mask(fwd, label):
    # Create a binary mask for the ROI
    # TODO: check that the mask contains at least one source
    src = fwd["src"]
    return get_label_mask(label, src)


def prepare_template(fwd, label, template):
    src = fwd["src"]

    # Support pre-defined options for template weights
    template = resolve_template(template, label, src)

    # Make sure w0 is a row vector with unit length
    template = np.atleast_1d(np.squeeze(template))
    if template.ndim > 1:
        raise ValueError(
            f"Template weights should be a vector, got "
            f"{template.ndim} dimensions instead"
        )
    template = template / norm(template)
    template = template[np.newaxis, :]


def prepare_inputs(fwd, label, template=None, source_cov=None):
    leadfield = prepare_leadfield(fwd)
    label_mask = prepare_label_mask(fwd, label)

    if template is not None:
        template = prepare_template(fwd, label, template)

    return leadfield, label_mask, template, source_cov
