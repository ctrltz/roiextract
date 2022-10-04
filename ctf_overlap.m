function [F, dF, Fadj, overlap_adj] = ctf_overlap(w, L, lambda)
%CTF_OVERLAP Diagonalizing dot product matrix of CTFs for N ROIs

    [n_rois, n_chans] = size(w);
    [~, n_voxels] = size(L);

    ctf = w * L;
    ctf2 = ctf .^ 2;
    overlap = ctf2 * ctf2';
    num = sum(diag(overlap));
    dnum = 4 * (ctf .^ 3) * L';
    denom = sum(overlap, 'all');

    ctf_tot = sum(ctf .^ 2, 1);
    mean_ctf_tot = mean(ctf_tot, 2);
    var_ctf_tot = var(ctf_tot, 0, 2);
    dmean_ctf_tot = 2 * ctf * L' / n_voxels;
    
    % could not make it better :(
    ddenom = zeros(n_rois, n_chans);
    dvar_ctf_tot = zeros(n_rois, n_chans);
    tmp = ctf * L';
    for m = 1:n_rois
        for n = 1:n_chans
            for j = 1:n_rois
                for k = 1:n_voxels
                    ddenom(m, n) = ddenom(m, n) + 4 * ctf(m, k) * ctf(j, k) * ctf(j, k) * L(n, k);
                end
            end

            for j = 1:n_voxels
                dvar_ctf_tot(m, n) = dvar_ctf_tot(m, n) + 4 / (n_voxels - 1) * (ctf_tot(j) - mean_ctf_tot) * (ctf(m, j) * L(n, j) - tmp(m, n) / n_voxels);
            end
        end
    end

    assert(isscalar(num));
    assert(isequal(size(w), size(dnum)));
    assert(isscalar(denom));
    assert(isequal(size(w), size(ddenom)));

    F1 = (-1) * num / denom;
    dF1 = (-1) * (dnum * denom - num * ddenom) / (denom ^ 2);
    F2 = var_ctf_tot / mean_ctf_tot;
    dF2 = (dvar_ctf_tot * mean_ctf_tot - var_ctf_tot * dmean_ctf_tot) / (mean_ctf_tot .^ 2);
    F = F1 + lambda * F2;
    dF = dF1 + lambda * dF2;
    Fadj = num / denom;
    overlap_adj = sqrt(overlap);
end

