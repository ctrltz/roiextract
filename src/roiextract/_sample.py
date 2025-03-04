from roiextract.optimize import ctf_optimize_label


def sample_criterion(lmbd, fwd, label, template, mode, dist_fun, return_filter=False):
    sf, props = ctf_optimize_label(
        fwd,
        label,
        template,
        lmbd,
        mode,
        initial="auto",
        quantify=True,
    )

    if return_filter:
        return sf

    x_key = "hom" if mode == "homogeneity" else "sim"
    dist = dist_fun(props[x_key], props["rat"])
    props_parts = [f"{k}: {v:.3g}" for k, v in props]
    props_desc = ", ".join(props_parts)
    print(f"sample_criterion | lambda={lmbd:.3g} | {props_desc} | dist={dist}")

    return dist
