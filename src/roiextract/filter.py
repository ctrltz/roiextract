import numpy as np
import mne

from .utils import (
    _check_input,
    data2stc,
    get_inverse_matrix,
    get_label_mask,
    get_aggregation_weights,
)


class SpatialFilter:
    """
    Convenience wrapper around a NumPy array that contains the spatial filter.

    Parameters:
    -----------
    w: array_like, shape (n_channels,)
        The weights of the spatial filter.
    method: str, optional
        Can be used to store the name of the method that was used to obtain the filter.
    method_params: dict, optional
        Can be used to store key parameters of the method for obtaining the filter.
    name: str, optional
        Can be used to add a unique name to the filter (e.g., name of the ROI).
    """

    def __init__(
        self,
        w,
        method="",
        method_params=dict(),
        name="",
    ) -> None:
        self.w = w
        self.method = method
        self.method_params = method_params
        self.name = name

    def __repr__(self) -> str:
        """
        Generate a short description for the filter in the following form:
        ([x] - only added if present)

        <SpatialFilter | [name] | [method (method_params)] | XX channels>
        """
        result = "<SpatialFilter"
        if self.name:
            result += f" | {self.name}"
        if self.method:
            params_str = [f"{k}={v}" for k, v in self.method_params.items()]
            params_str = ", ".join(params_str)
            params_str = f" ({params_str})" if params_str else ""
            result += f" | {self.method}{params_str}"
        result += f" | {self.w.size} channels>"

        return result

    @classmethod
    def from_inverse(
        cls,
        fwd,
        inv,
        label,
        inv_method,
        lambda2,
        roi_method,
        subject,
        subjects_dir,
    ):
        src = fwd["src"]
        mask = get_label_mask(label, src)
        W = get_inverse_matrix(inv, fwd, inv_method, lambda2)
        w_agg = get_aggregation_weights(
            roi_method, label, src, subject, subjects_dir
        )
        w = w_agg @ W[mask, :]
        return cls(
            np.atleast_1d(np.squeeze(w)),
            method=f"{inv_method}+{roi_method}",
            method_params=dict(lambda2=lambda2),
            name=label.name,
        )

    def apply(self, data) -> np.array:
        # TODO: check that the first dimension of data is suitable
        return self.w @ data

    def apply_raw(self, raw) -> np.array:
        return self.apply(raw.get_data())

    def get_ctf(
        self,
        L,
        mode="power",
        normalize="sum",
    ) -> np.array:
        _check_input("mode", mode, ["power", "amplitude"])
        _check_input("normalize", normalize, ["norm", "max", "sum", None])

        # Estimate the CTF
        ctf = self.w @ L
        if mode == "power":
            ctf = ctf**2

        # Normalize if needed
        if normalize == "norm":
            ctf /= np.linalg.norm(ctf)
        elif normalize == "max":
            ctf /= np.abs(ctf).max()
        elif normalize == "sum":
            ctf /= np.abs(ctf).sum()
        return ctf

    def get_ctf_fwd(
        self,
        fwd,
        mode="power",
        normalize="norm",
        subject=None,
    ) -> mne.SourceEstimate:
        leadfield = fwd["sol"]["data"]
        src = fwd["src"]
        return data2stc(
            self.get_ctf(leadfield, mode, normalize), src, subject=subject
        )

    def plot(self, info, **topomap_kwargs):
        # Make sure that the provided info has the correct amount of channels
        n_chans_filter = self.w.size
        n_chans_info = len(info["ch_names"])
        if n_chans_filter != n_chans_info:
            raise ValueError(
                f"The amount of channels in the provided Info object ({n_chans_info})"
                f" does not match the length of the spatial filter ({n_chans_filter})"
            )

        w = np.squeeze(self.w)
        return mne.viz.plot_topomap(w, info, **topomap_kwargs)


def apply_batch(data, filters) -> np.array:
    # TODO: check that all filters have the same number of channels
    # TODO: check that the first dimension of data is suitable
    w = np.vstack([sf.w[np.newaxis, :] for sf in filters])
    return w @ data


def apply_batch_raw(raw, filters) -> np.array:
    return apply_batch(raw.get_data(), filters)
