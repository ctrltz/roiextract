from functools import partial

import adaptive
import numpy as np

from roiextract._sample import AdaptiveMixin, BaseSampler
from roiextract.utils import normalize_values


def theta(xs, ys, limits):
    xs_norm, ys_norm = normalize_values(xs, ys, limits)

    # Normalize to [0, 1] range, make theta=0 match lambda=0
    return 1 - 2 * np.arctan(ys_norm / xs_norm) / np.pi


def dist_to_target_theta(sf, x_fun, y_fun, target, limits):
    x = x_fun(sf)
    y = y_fun(sf)
    return theta(x, y, limits) - target


class Suggester(BaseSampler, AdaptiveMixin):
    def __init__(
        self,
        filter_fun,
        param,
        eval_fun,
        bounds=(1e-5, 1.0 - 1e-5),
        sampling_limit=0.0001,
        **kwargs,
    ):
        super().__init__(
            filter_fun=filter_fun,
            param=param,
            bounds=bounds,
            eval_fun=eval_fun,
            sampling_limit=sampling_limit,
            **kwargs,
        )

    def _reset(self):
        super()._reset()
        self.limits = None
        self.crit_idx = None

    def _get_limits(self, x, y):
        self.sample_manually(self.bounds)
        xs = self.get(x)
        ys = self.get(y)

        _, _, limits = normalize_values(xs, ys, return_limits=True)
        return limits

    def suggest(self, crit_fun, loss_fun, pick_fun, tol, crit_key="crit"):
        self.eval_fun[crit_key] = crit_fun
        # TODO: use an ordered dict to avoid surprises
        self.crit_idx = -1

        learner = adaptive.Learner1D(
            self._sample,
            bounds=self.bounds,
            loss_per_interval=loss_fun,
        )
        adaptive.BlockingRunner(learner, loss_goal=tol, ntasks=1)

        param_values, eval_values = self._get_sampled_values(learner)
        crit_values = eval_values[crit_key]
        best_param = param_values[pick_fun(crit_values)]
        best_filter = self._get_filters([best_param])[0]

        return best_filter, best_param

    def suggest_for_theta(self, x, y, target_theta, tol=0.001):
        assert target_theta >= 0.0 and target_theta <= 1.0
        assert x in self.eval_fun
        assert y in self.eval_fun

        self.limits = self._get_limits(x, y)
        crit_fun = partial(
            dist_to_target_theta,
            x_fun=self.eval_fun[x],
            y_fun=self.eval_fun[y],
            target=target_theta,
            limits=self.limits,
        )

        return self.suggest(
            crit_fun=crit_fun,
            loss_fun=self._monotone_search_loss,
            pick_fun=lambda values: np.argmin(np.abs(values)),
            tol=tol,
        )

    def _monotone_search_loss(self, xs, ys):
        xs = np.array([x for x in xs if x is not None])
        ys = np.array([y for y in ys if y is not None])

        if ys.size < 2:
            return 0
        param_dist = np.abs(xs[1] - xs[0])
        if param_dist < self.sampling_limit:
            return 0

        # NOTE: If we sample a monotone function and the distance to the target
        # has opposite signs at the ends of an interval, we need to sample
        # another point in this interval. If the signs are the same, the
        # extremum does not belong to the interval, so we don't need to sample
        # this interval at all, setting the loss to 0.
        if ys[0][self.crit_idx] * ys[1][self.crit_idx] > 0:
            return 0

        return np.abs(ys).min()
