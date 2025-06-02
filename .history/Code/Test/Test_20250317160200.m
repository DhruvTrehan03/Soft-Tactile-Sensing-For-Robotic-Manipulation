clear;
run('Source/eidors-v3.11-ng/eidors/eidors_startup.m');  % Initialize EIDORS

%% Set variable height for extrusion (set H > 0 for a 3D extruded model, or H = 0 for 2D)
H = 0;  % Change to H > 0 for a 3D model

%% Define the 2D Shape (outer boundary and holes)
% Outer boundary (must be CLOCKWISE)
outer_xy = [0,   0;
            0,   3.6;
            4.4, 3.6;
            4.4, 0];

% Define the holes (also given in CLOCKWISE order per documentation)
hole1_xy = [1,   1;
            1,   2;
            1.5, 2;
            1.5, 1];

hole2_xy = [2.9, 1;
            2.9, 2;
            3.4, 2;
            3.4, 1];

%% Build the trunk_shape cell array using only the outer boundary.
% Format: { height, [x,y] (outer boundary), curve_type, maxsz }
trunk_shape = { H, {outer_xy,hole1_xy, hole2_xy}, 1, 0.2 };

%% Define Electrode Positions (if you do not need electrodes, leave empty)
elec_pos = [];

%% Define Electrode Shape (not used if elec_pos is empty)
elec_shape = 0.1;

%% Supply the holes via extra_ng_code
extra_ng_code = { hole1_xy, hole2_xy };

%% Create the Extruded Model
[fmdl, mat_idx] = ng_mk_extruded_model(trunk_shape, elec_pos, elec_shape);

%% Visualize the FEM Model
show_fem(fmdl);
title('Extruded Model: Outer Boundary with Excluded Holes');
