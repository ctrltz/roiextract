[cfg] = roiextract_config({'haufe', 'eeglab', 'tprod'});

load cm17;
load LEMON_sa_eLoreta.mat;
[n_chans, n_voxels] = size(L_normal);

ROI_inds = 1:96;
n_rois = numel(ROI_inds);

%% Calculate CTF ratio and dot product with a template CTF
ctf_r = zeros(n_rois, 2);
ctf_dp = zeros(n_rois, 2);
for i = 1:n_rois
    roi_ind = ROI_inds(i);

    I_in = zeros(1, n_voxels);
    voxels_roi = find(sa.cortex75K.in_HO(sa.voxels_5K_cort) == roi_ind);
    I_in(voxels_roi) = 1;

    x_avgflip = (signflip' .* I_in) * A_eloreta_normal';
    x_avg = I_in * A_eloreta_normal';
    w0_avg = ones(1, n_voxels) .* I_in; 
    w0_avgflip = signflip' .* I_in;

    [~, ~, ctf_r(i, 1)] = ctf_ratio(x_avgflip, L_normal, I_in);
    [~, ~, ctf_r(i, 2)] = ctf_ratio(x_avg, L_normal, I_in);
    [~, ~, ctf_dp(i, 1)] = ctf_dotprod_within(x_avgflip, L_normal, w0_avgflip, I_in);
    [~, ~, ctf_dp(i, 2)] = ctf_dotprod_within(x_avg, L_normal, w0_avg, I_in);
end

%% Display ratio and dotprod for all ROIs
ctf_r_voxels = zeros(numel(sa.cortex5K.in_from_cortex75K), 2);
ctf_dp_voxels = zeros(numel(sa.cortex5K.in_from_cortex75K), 2);
for i = 1:n_rois
    roi_ind = ROI_inds(i);

    I_in = zeros(1, n_voxels);
    voxels_roi = find(sa.cortex75K.in_HO(sa.voxels_5K_cort) == roi_ind);
    ctf_r_voxels(sa.cortex5K.in_cort(voxels_roi), :) = repmat(ctf_r(i, :), numel(voxels_roi), 1);
    ctf_dp_voxels(sa.cortex5K.in_cort(voxels_roi), :) = repmat(ctf_dp(i, :), numel(voxels_roi), 1);
end

ctf_diff = ctf_r(:, 1) - ctf_r(:, 2);
ctf_r_min = min(ctf_r, [], 'all');
ctf_r_max = max(ctf_r, [], 'all');
ctf_dp_min = min(ctf_dp, [], 'all');
ctf_dp_max = max(ctf_dp, [], 'all');

h = figure('Position', [400 250 840 420]);
allplots_cortex_subplots(sa, ctf_r_voxels(sa.cortex5K.in_to_cortex75K_geod, 1), [ctf_r_min ctf_r_max], cm17a, 'AVG-flip', 1, 'views', [5 1 3 0; 0 0 0 0]);
allplots_cortex_subplots(sa, ctf_r_voxels(sa.cortex5K.in_to_cortex75K_geod, 2), [ctf_r_min ctf_r_max], cm17a, 'AVG', 1, 'views', [0 0 0 0; 5 1 3 0]);
allplots_cortex_subplots(sa, ctf_r_voxels(sa.cortex5K.in_to_cortex75K_geod, 1), [ctf_r_min ctf_r_max], cm17a, 'AVG-flip & AVG', 1, 'views', [0 0 0 9; 0 0 0 0]);
sgtitle('CTF_{within} / CTF_{total}: AVG-flip | AVG');
exportgraphics(h, 'local/2022-09-07-compare-ctf-properties/ctf_ratio_avgflip_avg.png');

h = figure('Position', [400 250 840 420]);
allplots_cortex_subplots(sa, ctf_r_voxels(sa.cortex5K.in_to_cortex75K_geod, 1) - ...
    ctf_r_voxels(sa.cortex5K.in_to_cortex75K_geod, 2), [-max(abs(ctf_diff)) max(abs(ctf_diff))], ...
    cm17, 'DIFF', 1, 'views', [5 1 3 0]);
allplots_cortex_subplots(sa, ctf_r_voxels(sa.cortex5K.in_to_cortex75K_geod, 1) - ...
    ctf_r_voxels(sa.cortex5K.in_to_cortex75K_geod, 2), [-max(abs(ctf_diff)) max(abs(ctf_diff))], ...
    cm17, 'DIFF', 1, 'views', [0 0 0 9]);
sgtitle('CTF_{within} / CTF_{total}: AVG-flip > AVG');
exportgraphics(h, 'local/2022-09-07-compare-ctf-properties/ctf_ratio_contrast_avgflip_avg.png');

h = figure('Position', [400 250 840 420]);
allplots_cortex_subplots(sa, ctf_dp_voxels(sa.cortex5K.in_to_cortex75K_geod, 1), [ctf_dp_min ctf_dp_max], cm17a, 'AVG-flip', 1, 'views', [5 1 3 0; 0 0 0 0]);
allplots_cortex_subplots(sa, ctf_dp_voxels(sa.cortex5K.in_to_cortex75K_geod, 1), [ctf_dp_min ctf_dp_max], cm17a, 'AVG-flip', 1, 'views', [0 0 0 9; 0 0 0 0]);
allplots_cortex_subplots(sa, ctf_dp_voxels(sa.cortex5K.in_to_cortex75K_geod, 2), [ctf_dp_min ctf_dp_max], cm17a, 'AVG', 1, 'views', [0 0 0 0; 5 1 3 0]);
allplots_cortex_subplots(sa, ctf_dp_voxels(sa.cortex5K.in_to_cortex75K_geod, 2), [ctf_dp_min ctf_dp_max], cm17a, 'AVG', 1, 'views', [0 0 0 0; 0 0 0 9]);
sgtitle('CTF_{actual} * CTF_{ideal}: AVG-flip | AVG');
exportgraphics(h, 'local/2022-09-07-compare-ctf-properties/ctf_dotprod_avgflip_avg.png');

%% Compare methods for a particular ROI
roi_ind = 7; roi_name = 'precentral_gyrus_L';
% roi_ind = 17; roi_name = 'postcentral_gyrus_L';
% roi_ind = 48; roi_name = 'occipital_pole_L';
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
w0_lim = max(abs(w0));

options = optimoptions('fminunc', 'SpecifyObjectiveGradient', true, 'Display', 'final', ...
        'CheckGradients', true, 'MaxIterations', 1000, 'FiniteDifferenceType', 'central');

% CTF-based optimization (spatial filter)
alphas = [0 0.05 0.2 0.33 0.5 0.66 0.8 0.95 0.99 1];
n_alphas = numel(alphas);
dotprods = zeros(n_alphas, 1);
ratios = zeros(n_alphas, 1);
filters = zeros(n_alphas, n_chans);
for i = 1:n_alphas
    alpha = alphas(i);
    func = @(x) ctf_compromise(x, L_normal, w0, I_in, alpha);
    x_avgflip0 = fminunc(func, x_avgflip, options);

    [~, ~, dotprods(i), ratios(i)] = ctf_compromise(x_avgflip0, L_normal, w0, I_in, alpha);
    filters(i, :) = x_avgflip0;
end
patterns = filters * L_normal;

% CTF-based optimization (for an inverse model)
alphas = [0 0.05 0.2 0.33 0.5 0.66 0.8 0.95 0.99 1];
n_alphas = numel(alphas);
dotprods_inv = zeros(n_alphas, 1);
ratios_inv = zeros(n_alphas, 1);
filters_inv = zeros(n_alphas, numel(voxels_roi));
R = A_eloreta_normal' * L_normal;
R_roi = R(voxels_roi, :);
for i = 1:n_alphas
    alpha = alphas(i);
    func = @(x) ctf_compromise(x, R_roi, w0, I_in, alpha);
    w_avgflip0 = fminunc(func, w0_avgflip(voxels_roi), options);

    [~, ~, dotprods_inv(i), ratios_inv(i)] = ctf_compromise(w_avgflip0, R_roi, w0, I_in, alpha);
    filters_inv(i, :) = w_avgflip0;
end
patterns_inv = filters_inv * R_roi;

% More standard ways
filename = 'sub-010017_EO.set';
EEG = pop_loadset('filepath', cfg.path.lemon, 'filename', filename);
[EEG, EEG_narrow] = prepare_data(EEG, all_chanlocs, [8 13]);
A_lcmv_group = prepare_LCMV_inverse_operator({EEG.data, EEG_narrow.data}, L_normal, sa.myinds);
A_lcmv_bb = A_lcmv_group{1};
A_lcmv_nb = A_lcmv_group{2};

% eLoreta / bb / SVD
[~, ~, w_eLoreta_bb_svd] = sensor2roi(EEG.data, sa, A_eloreta_normal, 'svd', struct('n_comps', 1, 'roi_inds', roi_ind));
x_eLoreta_bb_svd = w_eLoreta_bb_svd{1}' * A_eloreta_normal(:, voxels_roi)';

% eLoreta / NB / SVD
[~, ~, w_eLoreta_nb_svd] = sensor2roi(EEG_narrow.data, sa, A_eloreta_normal, 'svd', struct('n_comps', 1, 'roi_inds', roi_ind));
x_eLoreta_nb_svd = w_eLoreta_nb_svd{1}' * A_eloreta_normal(:, voxels_roi)';

% LCMV / BB / SVD
[~, ~, w_lcmv_bb_svd] = sensor2roi(EEG.data, sa, A_lcmv_bb, 'svd', struct('n_comps', 1, 'roi_inds', roi_ind));
x_lcmv_bb_svd = w_lcmv_bb_svd{1}' * A_lcmv_bb(:, voxels_roi)';

% LCMV / NB / SVD
[~, ~, w_lcmv_nb_svd] = sensor2roi(EEG_narrow.data, sa, A_lcmv_nb, 'svd', struct('n_comps', 1, 'roi_inds', roi_ind));
x_lcmv_nb_svd = w_lcmv_nb_svd{1}' * A_lcmv_nb(:, voxels_roi)';

% eLoreta / AVG
x_eLoreta_avg = I_in * A_eloreta_normal';

% eLoreta / AVGflip
x_eLoreta_avgflip = (signflip' .* I_in) * A_eloreta_normal';

% eLoreta / Fidelity
x_eLoreta_fidelity = (weights .* I_in) * A_eloreta_normal';

% LCMV / BB / AVG
x_lcmv_bb_avg = I_in * A_lcmv_bb';

% LCMV / BB / AVGflip
x_lcmv_bb_avgflip = (signflip' .* I_in) * A_lcmv_bb';

% LCMV / NB / AVG
x_lcmv_nb_avg = I_in * A_lcmv_nb';

% LCMV / BB / AVGflip
x_lcmv_nb_avgflip = (signflip' .* I_in) * A_lcmv_nb';

% Merge the results
xs = [x_eLoreta_bb_svd; x_eLoreta_nb_svd; x_eLoreta_avg; x_eLoreta_avgflip; x_eLoreta_fidelity; ...
    x_lcmv_bb_svd; x_lcmv_bb_avg; x_lcmv_bb_avgflip; x_lcmv_nb_svd; x_lcmv_nb_avg; x_lcmv_nb_avgflip];
ctfs = xs * L_normal;
colors  = {'r' 'r' 'r' 'r' 'r' 'b' 'b' 'b' 'b' 'b' 'b'};
filled  = [1 0 1 1 1 1 1 1 0 0 0];
markers = {'o' 'o' '^' 's' 'pentagram' 'o' '^' 's' 'o' '^' 's'};
labels = {'eLoreta/bb/SVD', 'eLoreta/nb/SVD', 'eLoreta/AVG', 'eLoreta/AVG-flip', 'eLoreta/fidelity', ...
    'LCMV/bb/SVD', 'LCMV/bb/AVG', 'LCMV/bb/AVG-flip', 'LCMV/nb/SVD', 'LCMV/nb/avg', 'LCMV/nb/AVG-flip'};
n_datapoints = size(xs, 1);

dps = zeros(n_datapoints, 1);
rats = zeros(n_datapoints, 1);
for i = 1:n_datapoints
    [~, ~, dps(i), rats(i)] = ctf_compromise(xs(i, :), L_normal, w0, I_in, 0);
end

% Plot the results
h = figure('Position', [50 50 960 640]);
allplots_cortex_subplots(sa, cort5K2full(I_in, sa), [0 1], cm17a, 'mask', 1, 'views', [5 0 0; 0 0 0]);
allplots_cortex_subplots(sa, cort5K2full(w0, sa), [-w0_lim w0_lim], cm17, 'AVG-flip', 1, 'views', [0 0 0; 5 0 0]);
subplot(2, 3, [2 3 5 6]); hold on;
handles = [];
for i = 1:n_datapoints
    if filled(i)
        s2 = scatter(dps(i), rats(i), [], colors{i}, markers{i}, 'filled');
    else
        s2 = scatter(dps(i), rats(i), [], colors{i}, markers{i});
    end
    handles = [handles s2];
end
plot(dotprods, ratios, 'Color', [0.5 0.5 0.5], 'LineStyle', '--');
s_ctf = scatter(dotprods, ratios, [], alphas, '*');
handles = [handles s_ctf];
plot(dotprods_inv, ratios_inv, 'Color', 'r', 'LineStyle', '--');
s_ctfinv = scatter(dotprods_inv, ratios_inv, [], alphas, 'd');
handles = [handles s_ctfinv];
colormap(cm17);
xlabel('|Dot product with template CTF within ROI|');
ylabel('Ratio of CTF within ROI and total CTF');
axis equal; xlim([0 inf]); ylim([0 inf]);
legend(handles, [labels {'CTF-opt', 'CTF-eLoreta-opt'}], 'Location', 'southoutside', 'NumColumns', 3);
% legend(handles, labels, 'Location', 'southoutside', 'NumColumns', 3);
sgtitle([replace(roi_name, '_', ' ') ' | ' optimal_desc ' | Comparison']);
exportgraphics(h, ['local/2022-09-09-pipeline-comparison-ctf/' roi_name '_' optimal_desc '_comparison_wagg.png']);

% Plot the filters (sensor space)
h = figure;
h.WindowState = 'maximized';
pause(0.1);
for i = 1:n_datapoints
    subplot(3, 4, i);
    topoplot(xs(i, :), all_chanlocs);
    title(labels{i});
    cbar('vert', 0, get(gca, 'clim'));
end
subplot(3, 4, 11);
topoplot(filters(1, :), all_chanlocs);
title('Optimal Ratio');
cbar('vert', 0, get(gca, 'clim'));
subplot(3, 4, 12);
topoplot(filters(end, :), all_chanlocs);
title('Optimal Dotprod');
cbar('vert', 0, get(gca, 'clim'));
sgtitle([replace(roi_name, '_', ' ') ' | ' optimal_desc ' | Filter Weights']);
exportgraphics(h, ['local/2022-09-09-pipeline-comparison-ctf/' roi_name '_' optimal_desc '_filter.png']);

% Plot the optimized filters (sensor space)
h = figure;
h.WindowState = 'maximized';
filters = [filters; x_avgflip; w0_svd * A_eloreta_normal'];
labels = [cellfun(@(x) ['\alpha=', num2str(x, '%.2f')], ...
    num2cell(alphas), 'UniformOutput', false) {'AVG-flip', 'SVD-leadfield'}];
pause(0.1);
for i = 1:n_alphas+2
    subplot(3, 4, i);
    topoplot(filters(i, :), all_chanlocs);
    title(labels{i});
    cbar('vert', 0, get(gca, 'clim'));
end
sgtitle([replace(roi_name, '_', ' ') ' | ' optimal_desc ' | Filter Weights']);
exportgraphics(h, ['local/2022-09-09-pipeline-comparison-ctf/' roi_name '_' optimal_desc '_filters_opt.png']);

% Plot the optimized filters for an inverse model (sensor space)
h = figure;
h.WindowState = 'maximized';
filters_inv = [filters_inv * A_eloreta_normal(:, voxels_roi)'; x_avgflip; w0_svd * A_eloreta_normal'];
labels = [cellfun(@(x) ['\alpha=', num2str(x, '%.2f')], ...
    num2cell(alphas), 'UniformOutput', false) {'AVG-flip', 'SVD-leadfield'}];
pause(0.1);
for i = 1:n_alphas+2
    subplot(3, 4, i);
    topoplot(filters_inv(i, :), all_chanlocs);
    title(labels{i});
    cbar('vert', 0, get(gca, 'clim'));
end
sgtitle([replace(roi_name, '_', ' ') ' | ' optimal_desc ' | Filter Weights']);
exportgraphics(h, ['local/2022-09-09-pipeline-comparison-ctf/' roi_name '_' optimal_desc '_filters_inv_opt.png']);

% Plot the filters (ROI space)
h = figure;
h.WindowState = 'maximized';
pause(0.1);
filters_inv = [filters_inv; w0_avgflip(voxels_roi); w0_svd(voxels_roi)];
filters_inv = filters_inv ./ sum(abs(filters_inv), 2);
lim = max(abs(filters_inv), [], 2);
for i = 1:n_alphas+2
    filter_cort = zeros(numel(sa.voxels_5K_cort), 1);
    filter_cort(voxels_roi) = filters_inv(i, :);
    views = zeros(3, 4);
    views(floor((i - 1) / 4) + 1, mod(i - 1, 4) + 1) = 5;
    allplots_cortex_subplots(sa, cort5K2full(filter_cort, sa), [-lim(i) lim(i)], cm17, 'AVG-flip', 1, 'views', views);
    
    % zoom into the ROI
    vc_roi = sa.cortex75K.vc(sa.voxels_5K_cort(voxels_roi), [1 2]);
    ax = subplot(3, 4, i);
    ax.Visible = 'on';
    xlim([min(vc_roi(:, 1)) max(vc_roi(:, 1))]);
    ylim([min(vc_roi(:, 2)) max(vc_roi(:, 2))]);
    if i <= n_alphas
        xlabel(['\alpha = ', num2str(alphas(i))]);
    elseif i == (n_alphas + 1)
        xlabel('AVG-flip');
    else
        xlabel('SVD-leadfield');
    end
end
sgtitle([replace(roi_name, '_', ' ') ' | ' optimal_desc ' | ROI Filter Weights']);
exportgraphics(h, ['local/2022-09-09-pipeline-comparison-ctf/' roi_name '_' optimal_desc '_wagg.png']);

% Plot the patterns
h = figure;
h.WindowState = 'maximized';
pause(0.1);
ctfs = [ctfs; patterns(1, :); patterns(end-1, :)];
ctfs = ctfs ./ sum(abs(ctfs), 2);
lim = max(abs(ctfs), [], 2);
for i = 1:n_datapoints+2
    views = zeros(3, 4);
    views(floor((i - 1) / 4) + 1, mod(i - 1, 4) + 1) = 5;
    allplots_cortex_subplots(sa, cort5K2full(ctfs(i, :), sa), [-lim(i) lim(i)], cm17, 'AVG-flip', 1, 'views', views);
end
sgtitle([replace(roi_name, '_', ' ') ' | ' optimal_desc ' | CTF Patterns']);
exportgraphics(h, ['local/2022-09-09-pipeline-comparison-ctf/' roi_name '_' optimal_desc '_patterns.png']);