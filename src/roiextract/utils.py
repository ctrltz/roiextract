import logging
import numpy as np
import mne

from mne.label import label_sign_flip


logger = logging.getLogger("roiextract")


def _check_input(param, value, allowed_values):
    if value not in allowed_values:
        raise ValueError(f"Value {value} is not supported for {param}")


def _report_props(props):
    return ", ".join([f"{k}={v:.2g}" for k, v in props.items()])


def get_label_mask(label, src) -> np.ndarray:
    """
    Get a binary mask for vertices of the provided source space that
    belong to the specified ROI.

    Parameters
    ----------
    label : Label
        The ROI to consider.
    src : SourceSpaces
        The source space that defines all candidate source locations.

    Returns
    -------
    mask : array, shape (n_sources,)
        A vector with one logical value for each source in ``src``:
        True if it belongs to the ROI, False otherwise.
    """
    vertno = [s["vertno"] for s in src]
    nvert = [len(vn) for vn in vertno]
    if label.hemi == "lh":
        this_vertices = np.intersect1d(vertno[0], label.vertices)
        vert = np.searchsorted(vertno[0], this_vertices)
    elif label.hemi == "rh":
        this_vertices = np.intersect1d(vertno[1], label.vertices)
        vert = nvert[0] + np.searchsorted(vertno[1], this_vertices)
    else:
        raise ValueError("label %s has invalid hemi" % label.name)

    mask = np.zeros((sum(nvert),), dtype=int)
    mask[vert] = 1
    return mask > 0


def resolve_template(template, label, src):
    if isinstance(template, str):
        _check_input("template", template, ["mean_flip", "mean"])
        signflip = label_sign_flip(label, src)[np.newaxis, :]

        if template == "mean_flip":
            return signflip
        if template == "mean":
            return np.ones((1, signflip.size))

    return template


def data2stc(data, src, subject=None):
    vertno = [s["vertno"] for s in src]
    return mne.SourceEstimate(
        data=data, vertices=vertno, tmin=0, tstep=0.01, subject=subject
    )


def vertno_to_index(src, hemi, vertno):
    hemi_idx = ["lh", "rh"].index(hemi)
    index = np.searchsorted(src[hemi_idx]["vertno"], vertno).item()

    if hemi == "rh":
        index += src[0]["nuse"]

    return index


def normalize_values(xs, ys, limits=None, return_limits=False):
    if limits is not None:
        x_min, x_max, y_min, y_max = limits
    else:
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        limits = (x_min, x_max, y_min, y_max)

    xs_norm = (xs - x_min) / (x_max - x_min)
    ys_norm = (ys - y_min) / (y_max - y_min)

    if not return_limits:
        return xs_norm, ys_norm

    return xs_norm, ys_norm, limits
