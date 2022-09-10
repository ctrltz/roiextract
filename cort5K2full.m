function [data_remapped] = cort5K2full(data, sa)
%CORT5K2FULL Map data from 4502 voxels in cortex5K to cortex75K for visualization
    data = squeeze(data);
    data_remapped = zeros(numel(sa.cortex5K.in_from_cortex75K), 1); 
    assert(numel(data) == numel(sa.voxels_5K_cort));

    data_remapped(sa.cortex5K.in_cort) = data;
    data_remapped = data_remapped(sa.cortex5K.in_to_cortex75K_geod);
end

