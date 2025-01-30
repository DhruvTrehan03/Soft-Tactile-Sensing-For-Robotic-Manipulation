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
% 
% show_fem(mdl)

% Forward solvers $Id: forward_solvers01.m 3790 2013-04-04 15:41:27Z aadler $

% 2D Model
% imdl= mk_common_model('d2d1c',19);

% Create an homogeneous image
img_1 = mk_image(mdl, 1);
% h1= subplot(221);
% show_fem(img_1);

% Add a circular object at 0.2, 0.5
% Calculate element membership in object
img_2 = img_1;
select_fcn = inline('(x-0.2).^2+(y-0.5).^2<0.1^2','x','y','z');
h1= subplot(221);
show_fem(img_2);
img_2.elem_data = 1 + elem_select(img_2.fwd_model, select_fcn);
h2= subplot(222);
show_fem(img_2);

img_2.calc_colours.cb_shrink_move = [.3,.8,-0.02];
common_colourbar([h1,h2],img_2);

print_convert forward_solvers01a.png