roiextract_config({'haufe', 'eeglab'});

load cm17;
% load BCI_MI_sa_eLoreta.mat;
load LEMON_sa_eLoreta.mat;
[n_chans, n_voxels] = size(L_normal);

roi_ind = 7;  % precentral gyrus L
voxels_roi = find(sa.cortex75K.in_HO(sa.voxels_5K_cort) == roi_ind);
I_in = zeros(1, n_voxels);
I_in(voxels_roi) = 1;
w0_avgflip = signflip' .* I_in;
x_avgflip = (signflip' .* I_in) * A_eloreta_normal';
L_roi = L_normal(:, voxels_roi);
coeff = pca(L_roi);
w0_svd = zeros(1, n_voxels);
w0_svd(voxels_roi) = coeff(:, 1);
w0_svd_lim = max(abs(w0_svd));
L_in = L_normal(:, I_in > 0);
L_out = L_normal(:, I_in == 0);
L_mod = L_normal * (eye(n_voxels) - w0_avgflip' * w0_avgflip);

%% AVG-flip
ctf_avgflip = x_avgflip * L_normal;
lim = max(abs(ctf_avgflip));
% [~, ~, f_dp_avgflip, f_rat_avgflip] = ctf_compromise(x_avgflip, L_normal, w0_avgflip, I_in, 0)
[~, ~, f_dp_avgflip, f_rat_avgflip] = ctf_compromise(x_avgflip, L_normal, w0_svd, I_in, 0)
f_rat0_avgflip = ctf_ratio(x_avgflip, L_normal, I_in)

h = figure('Position', [400 250 840 420]);
% allplots_cortex_subplots(sa, cort5K2full(w0_avgflip, sa), [-1 1], cm17, 'ideal', 1, 'views', [5 0 0]);
allplots_cortex_subplots(sa, cort5K2full(w0_svd, sa), [-w0_svd_lim w0_svd_lim], cm17, 'ideal', 1, 'views', [5 0 0]);
allplots_cortex_subplots(sa, cort5K2full(ctf_avgflip, sa), [-lim lim], cm17, 'ctf', 1, 'views', [0 5 0]);
subplot(1, 3, 3); topoplot(x_avgflip, all_chanlocs); cbar('vert', 0, get(gca, 'clim'));
sgtitle({'AVG-flip', ...
    ['Dotprod: ' num2str(f_dp_avgflip, '%.4f')], ...
    ['Out/In Ratio^2: ' num2str(f_rat0_avgflip, '%.4f')], ...
    ['In/Total Ratio: ' num2str(f_rat_avgflip, '%.4f')]});
exportgraphics(h, 'local/2022-09-08-optimize-compromise/avgflip.png');

%% DeFleCT
% w_deflect = deflect(L_in, w0_avgflip(I_in > 0), L_mod, eye(n_chans), 0.05);
% w_deflect = deflect(L_in, w0_avgflip(I_in > 0), L_normal, eye(n_chans), 0.05);
% w_deflect = deflect(L_normal, w0_avgflip, L_normal, eye(n_chans), 0.05);
% w_deflect = deflect(L_in, w0_svd(I_in > 0), L_mod, eye(n_chans), 0.05);
% w_deflect = deflect(L_in, w0_svd(I_in > 0), L_normal, eye(n_chans), 0.05);
w_deflect = deflect(L_normal, w0_svd, L_normal, eye(n_chans), 0.5);
ctf_deflect = w_deflect * L_normal;
lim = max(abs(ctf_deflect));
[~, ~, f_dp_deflect, f_rat_deflect] = ctf_compromise(w_deflect, L_normal, w0_svd, I_in, 0)
f_rat0_deflect = ctf_ratio(w_deflect, L_normal, I_in)

h = figure('Position', [400 250 840 420]);
% allplots_cortex_subplots(sa, cort5K2full(w0_avgflip, sa), [-1 1], cm17, 'ideal', 1, 'views', [5 0 0]);
allplots_cortex_subplots(sa, cort5K2full(w0_svd, sa), [-w0_svd_lim w0_svd_lim], cm17, 'ideal', 1, 'views', [5 0 0]);
allplots_cortex_subplots(sa, cort5K2full(ctf_deflect, sa), [-lim lim], cm17, 'ctf', 1, 'views', [0 5 0]);
subplot(1, 3, 3); topoplot(w_deflect, all_chanlocs); cbar('vert', 0, get(gca, 'clim'));
sgtitle({'DeFleCT', ...
    ['Dotprod: ' num2str(f_dp_avgflip, '%.4f') ' -> ' num2str(f_dp_deflect, '%.4f')], ...
    ['Out/In Ratio^2: ' num2str(f_rat0_avgflip, '%.4f') ' -> ' num2str(f_rat0_deflect, '%.4f')], ...
    ['In/Total Ratio: ' num2str(f_rat_avgflip, '%.4f') ' -> ' num2str(f_rat_deflect, '%.4f')]});
% exportgraphics(h, 'local/2022-09-08-optimize-compromise/deflect_within_vs_rest.png');
% exportgraphics(h, 'local/2022-09-08-optimize-compromise/deflect_within_vs_total.png');
% exportgraphics(h, 'local/2022-09-08-optimize-compromise/deflect_total.png');

%% Compromise
alpha = 0.95;
func = @(x) ctf_compromise(x, L_normal, w0_avgflip, I_in, alpha);
options = optimoptions('fminunc', 'SpecifyObjectiveGradient', true, 'Display', 'iter', ...
    'CheckGradients', true, 'MaxIterations', 1000, 'FiniteDifferenceType', 'central');
x_avgflip0 = fminunc(func, x_avgflip, options);

[~, ~, f_dp_avgflip0, f_rat_avgflip0] = ctf_compromise(x_avgflip0, L_normal, w0_avgflip, I_in, alpha)
f_rat0_avgflip0 = ctf_ratio(x_avgflip0, L_normal, I_in)
ctf_avgflip0 = x_avgflip0 * L_normal;
lim = max(abs(ctf_avgflip0));

h = figure('Position', [400 250 840 420]);
allplots_cortex_subplots(sa, cort5K2full(w0_avgflip, sa), [-1 1], cm17, 'ideal', 1, 'views', [5 0 0]);
allplots_cortex_subplots(sa, cort5K2full(ctf_avgflip0, sa), [-lim lim], cm17, 'ctf', 1, 'views', [0 5 0]);
subplot(1, 3, 3); topoplot(x_avgflip0, all_chanlocs); cbar('vert', 0, get(gca, 'clim'));
sgtitle({['Optimize ' num2str(alpha, '%.2f') ' * Dotprod + ' num2str(1 - alpha, '%.2f') ' * Ratio'], ...
    ['Dotprod: ' num2str(f_dp_avgflip, '%.4f') ' -> ' num2str(f_dp_avgflip0, '%.4f')], ...
    ['Out/In Ratio^2: ' num2str(f_rat0_avgflip, '%.4f') ' -> ' num2str(f_rat0_avgflip0, '%.4f')], ...
    ['In/Total Ratio: ' num2str(f_rat_avgflip, '%.4f') ' -> ' num2str(f_rat_avgflip0, '%.4f')]});
exportgraphics(h, ['local/2022-09-08-optimize-compromise/ctf_compromise_avgflip0_alpha' num2str(alpha, '%.2f') '.png']);

%% Prepare and Save Filters for Regions of Interest
alphas = [0 0.05 0.2 0.33 0.5 0.66 0.8 0.95 0.99 1];
ROI_inds = [7 17 55 65];
options = optimoptions('fminunc', 'SpecifyObjectiveGradient', true, 'Display', 'final', ...
        'CheckGradients', true, 'MaxIterations', 1000, 'FiniteDifferenceType', 'central');
n_chans = size(L_normal, 1);
n_alphas = numel(alphas);
n_rois = numel(ROI_inds);
w_ctf = zeros(n_rois, n_chans, n_alphas);
ctf = zeros(n_rois, n_voxels, n_alphas);
dps = zeros(n_rois, n_chans, n_alphas);
rats = zeros(n_rois, n_chans, n_alphas);

for i_roi = 1:n_rois
    roi_ind = ROI_inds(i_roi);
    voxels_roi = find(sa.cortex75K.in_HO(sa.voxels_5K_cort) == roi_ind);
    I_in = zeros(1, n_voxels);
    I_in(voxels_roi) = 1;
    w0_avgflip = signflip' .* I_in;
    x_avgflip = (signflip' .* I_in) * A_eloreta_normal';
    L_roi = L_normal(:, voxels_roi);
    coeff = pca(L_roi);
    w0_svd = zeros(1, n_voxels);
    w0_svd(voxels_roi) = coeff(:, 1);
    % w0 = w0_avgflip; optimal_desc = 'AVG-flip';
    w0 = w0_svd; optimal_desc = 'SVD-leadfield';

    for i_alpha = 1:n_alphas
        alpha = alphas(i_alpha);
        func = @(x) ctf_compromise(x, L_normal, w0, I_in, alpha);
        x_avgflip0 = fminunc(func, x_avgflip, options);

        w_ctf(i_roi, :, i_alpha) = x_avgflip0;
        [~, ~, dps(i_roi, i_alpha), rats(i_roi, i_alpha)] = ctf_compromise(x_avgflip0, L_normal, w0, I_in, alpha);
    end
end

for i_alpha = 1:n_alphas
    ctf(:, :, i_alpha) = squeeze(w_ctf(:, :, i_alpha)) * L_normal;
end

% Plot filters
h = figure;
h.WindowState = 'maximized';
pause(0.1);
for i_roi = 1:n_rois
    for i_alpha = 1:n_alphas
        subplot(n_rois, n_alphas, (i_roi - 1) * n_alphas + i_alpha);
        topoplot(w_ctf(i_roi, :, i_alpha), all_chanlocs(sa.myinds));
        title([num2str(dps(i_roi, i_alpha), '%.3f') ' | ' num2str(rats(i_roi, i_alpha), '%.3f')]);
    end
end
sgtitle('Filter Weights | L->R: Ratio->Dotprod | T->B: preL-postL-preR-postR');
exportgraphics(h, 'local/2022-09-12-export-filters/filters_topomap.png');

% Plot patterns
h = figure;
h.WindowState = 'maximized';
pause(0.1);
lim = max(abs(ctf), [], 2);
for ir = 1:n_rois
    for ia = 1:n_alphas 
        views = zeros(n_rois, n_alphas);
        views(ir, ia) = 5;
        allplots_cortex_subplots(sa, cort5K2full(ctf(ir, :, ia), sa), [-lim(ir, ia) lim(ir, ia)], cm17, 'AVG-flip', 1, 'views', views);
    end
end
sgtitle('Filter Weights | L->R: Ratio->Dotprod | T->B: preL-postL-preR-postR');
exportgraphics(h, 'local/2022-09-12-export-filters/voxel_patterns.png');

save('local/BCI_MI_ctf_based_filters_sensorimotor.mat', 'w_ctf', 'alphas', 'ROI_inds');

%% Optimize several ROIs simultaneously
options = optimoptions('fminunc', 'SpecifyObjectiveGradient', true, 'Display', 'iter', ...
        'CheckGradients', false, 'MaxIterations', 1000, 'FiniteDifferenceType', 'central');

lambda = 100;
n_rois = 9;
func = @(x) ctf_overlap(x, L_normal, lambda);
x0 = zeros(n_rois, n_chans);
laplacian = [1 -0.25 -0.25 -0.25 -0.25];
x0(1, [33 1 21 32 4]) = laplacian;
x0(2, [34 2 21 35 6]) = laplacian;
x0(3, [8 36 41 40 44]) = laplacian;
x0(4, [11 39 42 43 47]) = laplacian;
x0(5, [50 14 18 19 24]) = laplacian;
x0(6, [20 51 52 56 47]) = laplacian;
x0(7, [17 44 48 49 53]) = laplacian;
x0(8, [58 23 28 57 59]) = laplacian;
x0(9, [60 25 30 59 61]) = laplacian;
x_opt = fminunc(func, x0, options);
[~, ~, ~, overlap] = ctf_overlap(x_opt, L_normal, lambda)
ctf_opt = x_opt * L_normal;

% Plot the optimized filters (sensor space)
h = figure;
h.WindowState = 'maximized';
pause(0.1);
for i = 1:n_rois
    subplot(3, 3, i);
%     topoplot(x_opt(i, :), all_chanlocs);
    topoplot(x0(i, :), all_chanlocs);
    cbar('vert', 0, get(gca, 'clim'));
end

% Plot the patterns
h = figure;
h.WindowState = 'maximized';
pause(0.1);
lim = max(abs(ctf_opt), [], 2);
for i = 1:n_rois
    views = zeros(3, 4);
    views(floor((i - 1) / 4) + 1, mod(i - 1, 4) + 1) = 5;
    allplots_cortex_subplots(sa, cort5K2full(ctf_opt(i, :), sa), [-lim(i) lim(i)], cm17, 'AVG-flip', 1, 'views', views);
end
