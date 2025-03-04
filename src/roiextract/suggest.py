import adaptive
import numpy as np

from copy import deepcopy
from functools import partial

from roiextract.inspect import OptimizationCurve
from roiextract.utils import _check_input, normalize
from roiextract._sample import sample_criterion


def theta(sims, rats, limits):
    sims_norm, rats_norm = normalize(sims, rats, limits)

    # Normalize to [0, 1] range, make theta=0 match lambda=0
    return 1 - 2 * np.arctan(rats_norm / sims_norm) / np.pi


def theta_dist(sims, rats, target, limits):
    return theta(sims, rats, limits) - target


class Suggester:
    def __init__(self, fwd, label, mode, template, sampling_limit=0.0001):
        _check_input("mode", mode, ["homogeneity", "similarity"])
        self.fwd = fwd
        self.label = label
        self.mode = mode
        self.template = template
        self.sampling_limit = sampling_limit
        self._reset()

    def _reset(self):
        self.filters = None
        self.lambdas = None
        self.n_points = None
        self._xs = None
        self._ys = None

    def copy(self):
        return deepcopy(self)

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

    @staticmethod
    def get_sampled_values(learner):
        lambdas = np.array(list(learner.data.keys()))
        sort_idx = np.argsort(lambdas)
        optim_curve = np.vstack(list(learner.data.values()))

        lambdas = lambdas[sort_idx]
        dist = optim_curve[sort_idx]

        return lambdas, dist

    def suggest_for_theta(self, target_theta, tol=0.001):
        assert target_theta >= 0.0 and target_theta <= 1.0

        oc = OptimizationCurve(self.fwd, self.label, self.mode, self.template)
        oc.sample_manually([0.0, 1.0])

        _, _, limits = normalize(oc.sims, oc.rats, return_limits=True)

        dist_fun = partial(theta_dist, target=target_theta, limits=limits)
        loss_fun = partial(self._monotone_search_loss)
        sample_fun = partial(
            sample_criterion,
            fwd=self.fwd,
            label=self.label,
            template=self.template,
            mode=self.mode,
            dist_fun=dist_fun,
        )

        learner = adaptive.Learner1D(
            sample_fun,
            bounds=(0, 1),
            loss_per_interval=loss_fun,
        )
        adaptive.BlockingRunner(learner, loss_goal=tol, ntasks=1)

        lambdas, dist = self.get_sampled_values(learner)
        best_lambda = lambdas[np.argmin(np.abs(dist))]

        oc.sample_manually([best_lambda])

        return best_lambda, oc.filters[0]

    def _monotone_search_loss(self, xs, ys):
        xs = np.array([x for x in xs if x is not None])
        ys = np.array([y for y in ys if y is not None])

        if ys.size < 2:
            return 0

        # TODO: take care of the sampling limit?
        # lambda_dist = np.abs(xs[1] - xs[0])

        # NOTE: If we sample a monotone function and the distance to the target
        # has opposite signs at the ends of an interval, we need to sample
        # another point in this interval. If the signs are the same, the
        # extremum does not belong to the interval, so we don't need to sample
        # this interval at all, setting the loss to 0.
        if ys[0] * ys[1] > 0:
            return 0

        return np.abs(ys).min()
