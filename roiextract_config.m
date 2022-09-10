function [] = roiextract_config(toolboxes)
%ROIEXTRACT_CONFIG Configuration for all paths that are necessary
    toolbox.base = '/data/p_02490/Toolboxes/';
    
    % Add precomputed matrices to path
    addpath('/data/p_02490/Scripts/neffb/precomputed/');

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
            otherwise
                error(['Unknown toolbox ', toolboxes{tb}]);
        end
        fprintf('OK\n');
    end   
end

