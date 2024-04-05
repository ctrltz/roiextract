import numpy as np
import numpy.typing as npt
import mne

from typing import Collection

from .utils import (
    _check_input,
    data2stc,
    get_inverse_matrix,
    get_label_mask,
    get_aggregation_weights,
)


class SpatialFilter:
    def __init__(
        self,
        w: npt.ArrayLike,
        method: str = "",
        method_params: dict = dict(),
        name: str = "",
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
            w,
            method=f"{inv_method}+{roi_method}",
            method_params=dict(lambda2=lambda2),
            name=label.name,
        )

    def apply(self, data: npt.ArrayLike) -> np.array:
        # TODO: check that the first dimension of data is suitable
        return self.w @ data

    def apply_raw(self, raw: mne.io.Raw | mne.io.RawArray) -> np.array:
        return self.apply(raw.get_data())

    def get_ctf(
        self,
        L: npt.ArrayLike,
        mode: str = "power",
        normalize: str | None = "sum",
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
        fwd: mne.Forward,
        mode: str = "power",
        normalize: str = "norm",
        subject: str | None = None,
    ) -> mne.SourceEstimate:
        leadfield = fwd["sol"]["data"]
        src = fwd["src"]
        return data2stc(
            self.get_ctf(leadfield, mode, normalize), src, subject=subject
        )

    def plot(self, info: mne.Info, **topomap_kwargs):
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


def apply_batch(data, filters: Collection[SpatialFilter]) -> np.array:
    # TODO: check that all filters have the same number of channels
    # TODO: check that the first dimension of data is suitable
    w = np.vstack([sf.w[np.newaxis, :] for sf in filters])
    return w @ data


def apply_batch_raw(raw, filters: Collection[SpatialFilter]) -> np.array:
    return apply_batch(raw.get_data(), filters)
