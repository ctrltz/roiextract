import adaptive
import numpy as np

from roiextract._sample import AdaptiveMixin, BaseSampler
from roiextract.suggest import Suggester
from roiextract.utils import logger, normalize_values


class OptimizationCurve(BaseSampler, AdaptiveMixin):
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
        self.mask = None

    @property
    def suggester(self):
        return Suggester(
            self.filter_fun,
            self.param,
            self.eval_fun.copy(),
            self.bounds,
            self.sampling_limit,
            **self.kwargs,
        )

    def _get_plot_values(self, x, y, normalize):
        xs, ys = self.get(x), self.get(y)
        if normalize:
            return normalize_values(xs, ys)

        return xs, ys

    def plot(self, x, y, ax, normalize=False, **kwargs):
        xs, ys = self._get_plot_values(x=x, y=y, normalize=normalize)
        ax.plot(xs, ys, **kwargs)

    def scatter(self, x, y, ax, normalize=False, **kwargs):
        xs, ys = self._get_plot_values(x=x, y=y, normalize=normalize)
        ax.scatter(xs, ys, c=self.param_values, **kwargs)

    def sample_adaptive(self, x, y, dist_threshold, reset=False):
        if reset:
            self._reset()

        self._prepare_mask(x, y)
        learner = adaptive.Learner1D(
            self._sample,
            bounds=self.bounds,
            loss_per_interval=self._optim_curve_loss,
        )
        adaptive.BlockingRunner(learner, loss_goal=dist_threshold, ntasks=1)

        param_values, eval_values = self._get_sampled_values(learner)
        filters = self._get_filters(param_values)
        self._update(param_values, eval_values, filters)

        return self

    def sample_homogeneous(self, n_points, reset=False):
        param_values = np.linspace(*self.bounds, num=n_points)

        return self.sample_manually(param_values, reset=reset)

    def save(self, filename):
        # Combine the data from all filters
        filters = np.vstack([sf.w for sf in self.filters])

        # Collect all the variables to save
        save_vars = dict(
            bounds=self.bounds,
            filters=filters,
            filter_fun=self.filter_fun.__name__,
            param=self.param,
            param_values=self.param_values,
            eval_keys=list(self.eval_fun.keys()),
        )
        for key in self.eval_fun:
            save_vars[key] = self.get(key)

        # Save
        logger.info(f"Saving the optimization curve to {filename}")
        np.savez(filename, **save_vars)

    def _prepare_mask(self, x, y):
        # XXX: current approach does not guarantee any order for x and y
        self.mask = np.full(len(self.eval_fun), False)
        keys = list(self.eval_fun.keys())
        self.mask[keys.index(x)] = True
        self.mask[keys.index(y)] = True

    def _optim_curve_loss(self, xs, ys):
        xs = [x for x in xs if x is not None]
        ys = [y for y in ys if y is not None]

        if len(ys) < 2:
            return 0
        param_dist = np.abs(xs[1] - xs[0])
        plane_dist = np.linalg.norm(ys[0][self.mask] - ys[1][self.mask])
        dist = np.maximum(param_dist, plane_dist)

        if param_dist < self.sampling_limit:
            return 0

        logger.debug(f"_optim_curve_loss | {ys} | {dist}")

        return dist
