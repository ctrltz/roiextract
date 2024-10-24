import adaptive
import numpy as np

from .optimize import ctf_optimize_label
from .utils import _check_input, logger


class OptimizationCurve:
    def __init__(self, fwd, label, mode, template):
        _check_input("mode", mode, ["homogeneity", "similarity"])
        self.fwd = fwd
        self.label = label
        self.mode = mode
        self.template = template
        self._reset()

    def _reset(self):
        self.filters = None
        self.lambdas = None
        self._xs = None
        self._ys = None

    @property
    def homs(self):
        if self.mode == "similarity":
            raise ValueError(
                f"The optimization mode is set to {self.mode}, so the "
                f"homogeneity values are not available."
            )

        return self._xs

    @property
    def sims(self):
        if self.mode == "homogeneity":
            raise ValueError(
                f"The optimization mode is set to {self.mode}, so the "
                f"similarity values are not available."
            )

        return self._xs

    @property
    def rats(self):
        return self._ys

    def plot_curve(self, ax):
        ax.plot(self._xs, self._ys, "k-")
        ax.scatter(self._xs, self._ys, c=self.lambdas)
        ax.set_aspect("equal")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel(f"CTF {self.mode}")
        ax.set_ylabel("CTF Ratio")

    def sample_adaptive(self, dist_threshold):
        self._reset()
        self.filters = []
        learner = adaptive.Learner1D(
            self._sample,
            bounds=(0, 1),
            loss_per_interval=self._optim_curve_loss,
        )
        adaptive.BlockingRunner(learner, loss_goal=dist_threshold, ntasks=1)

        self._get_sampled_values(learner)
        self._get_filters()

    def sample_homogeneous(self, n_points):
        self._reset()
        self.filters = []
        self.lambdas = np.linspace(0, 1, num=n_points)
        self.xs = np.zeros((n_points,))
        self.ys = np.zeros((n_points,))

        for i, lmbd in enumerate(self.lambdas):
            result = self._sample(lmbd)
            self.xs[i], self.ys[i] = result
        self._get_filters()

        return self

    def save(self, filename):
        # Combine the data from all filters
        filters = np.vstack([sf.w for sf in self.filters])

        # Collect all the variables to save
        save_vars = dict(filters=filters, lambdas=self.lambdas, rats=self.rats)
        if self.mode == "homogeneity":
            save_vars["homs"] = self.homs
        else:
            save_vars["sims"] = self.sims

        # Save
        logger.info("Saving the optimization curve to {filename}")
        np.savez(filename, **save_vars)

    def _sample(self, lmbd, return_filter=False):
        sf, props = ctf_optimize_label(
            self.fwd,
            self.label,
            self.template,
            lmbd,
            mode=self.mode,
            initial="auto",
            quantify=True,
        )
        logger.debug(f"_sample | lambda={lmbd:.3g} | {props}")

        if return_filter:
            return sf

        x_key = "hom" if self.mode == "homogeneity" else "sim"
        return np.array([props[x_key], props["rat"]])

    def _get_filters(self):
        self.filters = [
            self._sample(lmbd, return_filter=True) for lmbd in self.lambdas
        ]

    def _get_sampled_values(self, learner):
        lambdas = np.array(list(learner.data.keys()))
        sort_idx = np.argsort(lambdas)
        optim_curve = np.vstack(list(learner.data.values()))

        self.lambdas = lambdas[sort_idx]
        self._xs = optim_curve[sort_idx, 0]
        self._ys = optim_curve[sort_idx, 1]

    def _optim_curve_loss(self, xs, ys):
        xs = [x for x in xs if x is not None]
        ys = [y for y in ys if y is not None]

        if len(ys) < 2:
            return 0
        dist = np.linalg.norm(ys[0] - ys[1])

        logger.debug(f"_optim_curve_loss | {ys} | {dist}")

        return dist
