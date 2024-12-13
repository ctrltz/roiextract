import numpy as np
import mne
import warnings

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
    ch_names: list or None, optional
        Can be used to store names of the channels that the weights correspond to.
    name: str, optional
        Can be used to add a unique name to the filter (e.g., name of the ROI).
    """

    def __init__(
        self,
        w,
        method="",
        method_params=dict(),
        ch_names=None,
        name="",
    ) -> None:
        self.w = np.squeeze(np.array(w))
        self.method = method
        self.method_params = method_params
        self._validate_ch_names(ch_names)
        self.name = name

    @property
    def size(self):
        return self.w.size

    def _validate_ch_names(self, ch_names):
        self.ch_names = None
        if ch_names is None:
            return

        if self.size != len(ch_names):
            raise ValueError(
                "The number of channel names should correspond to the number "
                "of provided weights"
            )

        if len(set(ch_names)) != len(ch_names):
            raise ValueError("All channel names should be unique.")

        self.ch_names = np.array(ch_names)

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
        result += f" | {self.size} channels>"

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
        ch_names = fwd["info"]["ch_names"]
        mask = get_label_mask(label, src)
        W = get_inverse_matrix(inv, fwd, inv_method, lambda2)
        w_agg = get_aggregation_weights(roi_method, label, src, subject, subjects_dir)
        w = w_agg @ W[mask, :]
        return cls(
            np.atleast_1d(np.squeeze(w)),
            method=f"{inv_method}+{roi_method}",
            method_params=dict(lambda2=lambda2),
            ch_names=ch_names,
            name=label.name,
        )

    def _align(self, num_channels, raw_names):
        print(f"{self.ch_names=}")
        print(f"{raw_names=}")
        has_own_names = self.ch_names is not None
        has_raw_names = raw_names is not None
        alignment_possible = has_own_names and has_raw_names

        # Warn if alignment is not possible
        if not alignment_possible:
            warnings.warn(
                "The filter or the provided data object does not contain the "
                "information about channel names. "
                "To obtain correct results, please ensure that the order of "
                "the channels is the same for the filter and the M/EEG data."
            )

        # Check that the number of channels in raw and filter matches
        if num_channels != self.size:
            raise ValueError(
                "The number of channels in the provided data object is "
                "different from the number of channels in the filter, "
                "can't proceed."
            )

        # If there is no information to reorder channels, use the original order
        if not alignment_possible:
            return np.arange(self.size)

        # Otherwise, align the order of channels to match the provided data object
        common, ind1, ind2 = np.intersect1d(
            self.ch_names, raw_names, return_indices=True
        )
        if len(common) != len(raw_names):
            missing = set(self.ch_names) - set(raw_names)
            raise ValueError(
                f"The following channels are required to apply the filter but "
                f"aren't present in the provided data object: {', '.join(missing)}. "
            )

        mapping = np.zeros((self.size,), dtype=int)
        mapping[ind2] = ind1

        return mapping

    def apply(self, data, ch_names=None) -> np.array:
        reorder = self._align(data.shape[0], ch_names)
        return self.w[np.newaxis, reorder] @ data

    def apply_raw(self, raw) -> np.array:
        return self.apply(raw.get_data(), raw.ch_names)

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
        return data2stc(self.get_ctf(leadfield, mode, normalize), src, subject=subject)

    def plot(self, info, **topomap_kwargs):
        # Make sure that the provided info has the correct amount of channels
        n_chans_filter = self.size
        n_chans_info = len(info["ch_names"])
        if n_chans_filter != n_chans_info:
            raise ValueError(
                f"The amount of channels in the provided Info object ({n_chans_info})"
                f" does not match the length of the spatial filter ({n_chans_filter})"
            )

        w = np.squeeze(self.w)
        return mne.viz.plot_topomap(w, info, **topomap_kwargs)


def apply_batch(data, filters, ch_names=None) -> np.array:
    weights = []
    for sf in filters:
        reorder = sf._align(data.shape[0], ch_names)
        weights.append(sf.w[np.newaxis, reorder])
    w = np.vstack(weights)
    return w @ data


def apply_batch_raw(raw, filters) -> np.array:
    return apply_batch(raw.get_data(), filters, raw.ch_names)
