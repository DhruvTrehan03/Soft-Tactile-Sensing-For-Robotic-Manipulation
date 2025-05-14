clear;
run('Source/eidors-v3.11-ng/eidors/eidors_startup.m'); % Initialize EIDORS

% Define Model Parameters
height = 0;
width = 4.4;
len = 3.6;
xy = [0 0 width width; 0 len len 0]';
curve_type = 1;
maxsz = 0.1;
trunk_shape = {height, xy, curve_type, maxsz};

% Define Electrodes
elec_pos  = [32, 1.1];
elec_shape = 0.2;

% Generate Model
mdl = ng_mk_extruded_model(trunk_shape, elec_pos, elec_shape);
stim = mk_stim_patterns(32, 1, [0, 16], [0, 1], {'no_meas_current'}, 5);
plain = mk_image(mdl, 1, 'Hi');
plain.fwd_model.stimulation = stim;

save_path = 'Code/Analysis/ModelGen.mat';
save(save_path, 'mdl', 'stim', 'plain');