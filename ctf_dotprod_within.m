function [F, dF, Fadj] = ctf_dotprod_within(w, L, w0, I_in)
%CTF_DOTPROD Dot product of desired and constructed filters
    % Use only voxels within the mask
    w0_in = w0(I_in > 0);
    L_in = L(:, I_in > 0);

    % Make sure w0 is unit length
    w0_in = w0_in ./ norm(w0_in);

    dotprod = w * L_in * w0_in';
    ctf = w * L_in;

    assert(isscalar(dotprod));
    assert(isrow(ctf));

    F = (-1) * (dotprod ^ 2) / (norm(ctf) ^ 2);
    dF = (-2) * w * (F * (L_in * L_in') + L_in * (w0_in' * w0_in) * L_in') / (norm(ctf) ^ 2);

    Fadj = dotprod / norm(ctf);
end

