function [F, dF, F_dp_adj, F_rat_adj] = ctf_compromise(w, L, w0, I_in, alpha)
%CTF_COMPROMISE Optimize dotprod within ROI and in/out ratio simultaneously
    [F_dp, dF_dp, F_dp_adj] = ctf_dotprod_within(w, L, w0, I_in);
    [F_rat, dF_rat, F_rat_adj] = ctf_ratio(w, L, I_in);

    F = alpha * F_dp + (1 - alpha) * F_rat;
    dF = alpha * dF_dp + (1 - alpha) * dF_rat;
end

