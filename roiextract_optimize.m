roiextract_config({'haufe', 'eeglab'});

load cm17;
load LEMON_sa_eLoreta.mat;
[n_chans, n_voxels] = size(L_normal);

roi_ind = 7;  % precentral gyrus L
voxels_roi = find(sa.cortex75K.in_HO(sa.voxels_5K_cort) == roi_ind);
I_in = zeros(1, n_voxels);
I_in(voxels_roi) = 1;
w0_avgflip = signflip' .* I_in;
x_avgflip = (signflip' .* I_in) * A_eloreta_normal';
L_in = L_normal(:, I_in > 0);
L_mod = L_normal * (eye(n_voxels) - w0_avgflip' * w0_avgflip);

%% AVG-flip
ctf_avgflip = x_avgflip * L_normal;
lim = max(abs(ctf_avgflip));
[~, ~, f_dp_avgflip, f_rat_avgflip] = ctf_compromise(x_avgflip, L_normal, w0_avgflip, I_in, alpha)
f_rat0_avgflip = ctf_ratio(x_avgflip, L_normal, I_in)

h = figure('Position', [400 250 840 420]);
allplots_cortex_subplots(sa, cort5K2full(w0_avgflip, sa), [-1 1], cm17, 'ideal', 1, 'views', [5 0 0]);
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
w_deflect = deflect(L_normal, w0_avgflip, L_normal, eye(n_chans), 0.05);
ctf_deflect = w_deflect * L_normal;
lim = max(abs(ctf_deflect));
[~, ~, f_dp_deflect, f_rat_deflect] = ctf_compromise(w_deflect, L_normal, w0_avgflip, I_in, alpha)
f_rat0_deflect = ctf_ratio(w_deflect, L_normal, I_in)

h = figure('Position', [400 250 840 420]);
allplots_cortex_subplots(sa, cort5K2full(w0_avgflip, sa), [-1 1], cm17, 'ideal', 1, 'views', [5 0 0]);
allplots_cortex_subplots(sa, cort5K2full(ctf_deflect, sa), [-lim lim], cm17, 'ctf', 1, 'views', [0 5 0]);
subplot(1, 3, 3); topoplot(w_deflect, all_chanlocs); cbar('vert', 0, get(gca, 'clim'));
sgtitle({'DeFleCT', ...
    ['Dotprod: ' num2str(f_dp_avgflip, '%.4f') ' -> ' num2str(f_dp_deflect, '%.4f')], ...
    ['Out/In Ratio^2: ' num2str(f_rat0_avgflip, '%.4f') ' -> ' num2str(f_rat0_deflect, '%.4f')], ...
    ['In/Total Ratio: ' num2str(f_rat_avgflip, '%.4f') ' -> ' num2str(f_rat_deflect, '%.4f')]});
% exportgraphics(h, 'local/2022-09-08-optimize-compromise/deflect_within_vs_rest.png');
% exportgraphics(h, 'local/2022-09-08-optimize-compromise/deflect_within_vs_total.png');
exportgraphics(h, 'local/2022-09-08-optimize-compromise/deflect_total.png');

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