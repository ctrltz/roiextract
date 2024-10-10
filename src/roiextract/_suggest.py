from .utils import logger, _report_props


INITIAL_LAMBDAS = {"rat": 0, "sim": 0.999, "hom": 0.999}
IS_DECREASING = {"rat": True, "sim": False, "hom": False}


def suggest_lambda(opt_func, quant_func, criteria, threshold, tol=0.001):
    initial_lambda = INITIAL_LAMBDAS[criteria]
    props = quant_func(w=opt_func(lambda_=initial_lambda))
    crit_thresh = threshold * props[criteria]
    logger.info(
        f"Suggesting lambda to obtain at least {threshold:.2g} of max {criteria}"
    )
    logger.info(f"Properties (lambda={initial_lambda}): {_report_props(props)}")
    logger.info(f"Criteria threshold: {crit_thresh:.3g}")

    left, right = 0, 1
    while right - left > tol:
        mid = (left + right) / 2
        props = quant_func(w=opt_func(lambda_=mid))
        logger.info(f"Properties (lambda={mid:.3g}): {_report_props(props)}")
        if IS_DECREASING[criteria]:
            if props[criteria] > crit_thresh:
                left = mid
            else:
                right = mid
        else:
            if props[criteria] > crit_thresh:
                right = mid
            else:
                left = mid

    return (left + right) / 2
