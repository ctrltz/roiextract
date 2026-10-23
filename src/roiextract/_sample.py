from copy import deepcopy

import numpy as np

from roiextract.utils import logger


class BaseSampler:
    def __init__(self, filter_fun, param, bounds, eval_fun, sampling_limit, **kwargs):
        self.filter_fun = filter_fun
        self.param = param
        self.bounds = bounds
        self.kwargs = kwargs
        self.eval_fun = eval_fun
        self.sampling_limit = sampling_limit
        self._reset()

    def _reset(self):
        self.param_values = None
        self.eval_values = None
        self.filters = None

    def copy(self):
        return deepcopy(self)

    def get(self, key):
        return self.eval_values[key]

    def __getitem__(self, key):
        param_value = key
        if isinstance(key, tuple):
            param_value, eval_key = key

        index = np.nonzero(self.param_values == param_value)[0]
        if not index.size:
            raise ValueError(f"Parameter value {param_value} was not sampled")

        if eval_key is not None:
            return self.eval_values[eval_key][index]

        return {k: v[index] for k, v in self.eval_values.items()}

    def _update(self, param_values, eval_values, filters):
        if self.param_values is None:
            self.param_values = param_values
            self.eval_values = eval_values
            self.filters = filters
            return

        insert_idx = np.searchsorted(self.param_values, param_values)
        self.param_values = np.insert(self.param_values, insert_idx, param_values)
        self.filters = np.insert(self.filters, insert_idx, filters)
        for key in self.eval_fun:
            self.eval_values[key] = np.insert(
                self.eval_values[key], insert_idx, eval_values[key]
            )

    def sample_manually(self, param_values, reset=False):
        if reset:
            self._reset()

        n_pv = len(param_values)
        eval_values = {key: np.full(n_pv, np.nan) for key in self.eval_fun}

        # TODO: add parallelization?
        for i, pv in enumerate(param_values):
            for key, v in zip(self.eval_fun, self._sample(pv)):
                eval_values[key][i] = v

        filters = self._get_filters(param_values)
        self._update(param_values, eval_values, filters)

        return self

    def _sample(self, value, return_filter=False):
        sf = self.filter_fun(**self._prepare_kwargs(value))
        logger.debug(f"_sample | {self.param}={value:.3g}")

        if return_filter:
            return sf

        return np.array([fun(sf) for fun in self.eval_fun.values()])

    def _prepare_kwargs(self, value):
        kwargs = self.kwargs.copy()
        kwargs[self.param] = value

        return kwargs

    def _get_filters(self, param_values):
        return np.array([self._sample(pv, return_filter=True) for pv in param_values])


class AdaptiveMixin:
    def _get_sampled_values(self, learner):
        param_values = np.array(list(learner.data.keys()))
        sort_idx = np.argsort(param_values)
        optim_curve = np.vstack(list(learner.data.values()))

        param_values = param_values[sort_idx]
        eval_values = {}
        for idx, key in enumerate(self.eval_fun):
            eval_values[key] = optim_curve[sort_idx, idx]

        return param_values, eval_values
