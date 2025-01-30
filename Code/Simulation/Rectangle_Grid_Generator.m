run eidors-v3.11-ng\eidors\eidors_startup.m

% Parameters that define the grid
x_min = 0; % Minimum x value
x_max = 1; % Maximum x value
y_min = 0; % Minimum y value
y_max = 1; % Maximum y value
x_points = 80;
y_points = 100;
x_elec = 8; % Number of x electrodes (including ends) on the bottom and top rows
y_elec = 10; % Number of y electrodes (including ends) on the left and right columns

% Generate the x-coordinates (moving horizontally)
x1 = linspace(x_min, x_max, x_points)'; % Left to right on bottom (from 0 to 1 with step 0.1)
x2 = linspace(x_max, x_min, x_points)'; % Right to left on top

% Generate the y-coordinates (moving vertically)
y1 = y_min * ones(size(x1)); % Bottom row (all zeros)
y2 = linspace(y_min, y_max, y_points)'; % Right vertical from bottom to top (increment 0.125)
y3 = y_max * ones(size(x2)); % Top row (all ones)
y4 = linspace(y_max, y_min, y_points)'; % Left vertical from top to bottom (increment 0.125)

xy = [
    [x1, y1]; % Bottom horizontal line
    [x_max * ones(size(y2(2:end))), y2(2:end)]; % Right vertical line, excluding bottom-right corner
    [x2(2:end), y3(2:end)]; % Top horizontal line, excluding top-right corner
    [x_min * ones(size(y4(2:end-1))), y4(2:end-1)]; % Left vertical line, excluding bottom-left and top-left corners
];

ex1 = linspace(x_min, x_max, x_elec)'; % Left to right on bottom (from 0 to 1 with step 0.1)
ex2 = linspace(x_max, x_min, x_elec)'; % Right to left on top

% Generate the y-coordinates (moving vertically)
ey1 = y_min * ones(size(ex1)); % Bottom row (all zeros)
ey2 = linspace(y_min, y_max, y_elec)'; % Right vertical from bottom to top (increment 0.125)
ey3 = y_max * ones(size(ex2)); % Top row (all ones)
ey4 = linspace(y_max, y_min, y_elec)'; % Left vertical from top to bottom (increment 0.125)
% Combine the sections into a single matrix for [x y] without repeating points at corners

exy = [
    [ex1, ey1]; % Bottom horizontal line
    [x_max * ones(size(ey2(2:end))), ey2(2:end)]; % Right vertical line, excluding bottom-right corner
    [ex2(2:end), ey3(2:end)]; % Top horizontal line, excluding top-right corner
    [x_min * ones(size(ey4(2:end-1))), ey4(2:end-1)]; % Left vertical line, excluding bottom-left and top-left corners
];
% Display the result
% disp(xy);

elec_shape = 0.4;

mdl = ng_mk_2d_model(xy, exy);
img = mk_image(mdl, 1);

show_fem(mdl)

% xy2 = [0 0;  0.1 0; 0.2 0; 0.3 0; 0.4 0; 0.5 0; 0.6 0; 0.7 0; 0.8 0; 0.9 0; 1 0; 
%      1 0.125; 1 0.25;  1 0.375;  1 0.5;  1 0.625;  1 0.75;  1 0.875;  1 1;  
%      0.9 1;  0.8 1;  0.7 1;  0.6 1;  0.5 1;  0.4 1;  0.3 1;  0.2 1;  0.1 1;  0 1;
%      0 0.875; 0 0.75; 0 0.625; 0 0.5; 0 0.375; 0 0.25;  0 0.125;];
% 
% disp(xy2)