import logging
import numpy as np
import mne


logger = logging.getLogger("roiextract")


def _check_input(param, value, allowed_values):
    if value not in allowed_values:
        raise ValueError(f"Value {value} is not supported for {param}")


def _report_props(props):
    return ", ".join([f"{k}={v:.2g}" for k, v in props.items()])


def get_label_mask(label, src):
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
        from mne.label import label_sign_flip

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


def get_inverse_matrix(inv, fwd, method, lambda_):
    # TODO: get rid of private dependency
    from mne.minimum_norm.resolution_matrix import (
        _get_matrix_from_inverse_operator,
    )

    return _get_matrix_from_inverse_operator(inv, fwd, method, lambda2=lambda_)


def get_aggregation_weights(method, label, src, subject, subjects_dir):
    _check_input("method", method, ["mean", "mean_flip", "centroid"])
    if method in ["mean", "mean_flip"]:
        return resolve_template(method, label, src)

    # find the center of mass
    hemi_idx = 0 if label.hemi == "lh" else 1
    label_vertices = label.get_vertices_used(src[hemi_idx]["vertno"])
    centroid_idx = label.center_of_mass(
        subject=subject, restrict_vertices=src, subjects_dir=subjects_dir
    )
    return (label_vertices == centroid_idx).astype(int)
