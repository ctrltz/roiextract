import numpy as np

from roiextract.filter import SpatialFilter
from roiextract.inspect import OptimizationCurve


def load_optimization_curve(filename):
    oc = OptimizationCurve(None, None, "similarity", None)
    with np.load(filename) as data:
        oc.lambdas = data["lambdas"]
        oc.filters = [
            SpatialFilter(w=w, method_params=dict(lambda_=lmbd))
            for w, lmbd in zip(data["filters"], oc.lambdas)
        ]
        oc.n_points = len(oc.lambdas)
        oc._ys = data["rats"]
        oc._xs = data.get("homs", data["sims"])

    return oc
