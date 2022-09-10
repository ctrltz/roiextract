roiextract_config({'haufe', 'eeglab'});

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

%%
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