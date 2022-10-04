function [cfg] = roiextract_config(toolboxes)
%ROIEXTRACT_CONFIG Configuration for all paths that are necessary
    path.lemon = '/data/pt_nro109/Share/EEG_MPILMBB_LEMON/EEG_Preprocessed_BIDS_ID/EEG_Preprocessed/';

    toolbox.base = '/data/p_02490/Toolboxes/';
    
    % Add precomputed matrices and inverse modeling to path
    addpath('/data/p_02490/Scripts/neffb/precomputed/');
    addpath('/data/p_02490/Scripts/neffb/inverse/');
    addpath('/data/p_02490/Scripts/neffb/');

    % Load toolboxes
    n_toolboxes = numel(toolboxes);
    for tb = 1:n_toolboxes
        fprintf('Loading toolbox %s...\n', toolboxes{tb});
        switch (toolboxes{tb})
            case 'eeglab'
                toolbox.eeglab = [toolbox.base 'eeglab2021.0/'];
                addpath(toolbox.eeglab);
                eeglab;
            case 'haufe'
                toolbox.haufe = [toolbox.base 'haufe/'];
                addpath(toolbox.haufe);
                load cm17;
            case 'tprod'
                % NOTE: tprod tests passed only when everything was in double precision
                toolbox.tprod = [toolbox.base 'tprod/'];
                addpath(genpath(toolbox.tprod));
            otherwise
                error(['Unknown toolbox ', toolboxes{tb}]);
        end
        fprintf('OK\n');
    end   

    cfg.path = path;
end

