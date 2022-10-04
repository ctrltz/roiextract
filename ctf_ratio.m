function [F, dF, Fadj] = ctf_ratio(w, L, I_in)
%CTF_RATIO Ratio of squared CTF outside and within ROI
    % Split lead field matrix into in and out parts
    L_in = L(:, I_in > 0);
    L_out = L(:, I_in == 0);

    ctf = w * L;
    ctf_out = w * L_out;
    ctf_in = w * L_in;

    assert(isrow(ctf_out));
    assert(isrow(ctf_in));
    
    F_out = norm(ctf_out) ^ 2;
    F_in = norm(ctf_in) ^ 2;

    F = F_out / F_in;
    dF = 2 * w * (L_out * L_out' - F * (L_in * L_in')) / F_in;

    Fadj = norm(ctf_in) / norm(ctf);
end

