run eidors-v3.11-ng\eidors\eidors_startup.m

% Parameters that define the grid
x_min = 0; % Minimum x value
x_max = 1; % Maximum x value
y_min = 0; % Minimum y value
y_max = 1.25; % Maximum y value
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

exy = [
    [ex1, ey1]; % Bottom horizontal line
    [x_max * ones(size(ey2(2:end))), ey2(2:end)]; % Right vertical line, excluding bottom-right corner
    [ex2(2:end), ey3(2:end)]; % Top horizontal line, excluding top-right corner
    [x_min * ones(size(ey4(2:end-1))), ey4(2:end-1)]; % Left vertical line, excluding bottom-left and top-left corners
];

% Model and stimulation pattern
mdl = ng_mk_2d_model(xy, exy);
img_1 = mk_image(mdl, 1);
stim = mk_stim_patterns(32, 1, '{op}', '{ad}');
img_1.fwd_model.stimulation = stim;

% Initial plot setup
h1 = subplot(221);
show_fem(img_1);

% Homogeneous data computation for reference
h2 = subplot(222);
homg_data = fwd_solve(img_1);
xax2 = 1:length(homg_data.meas);
plot(xax2, homg_data.meas);
title('Homogeneous Measurement Data');
xlabel('Measurement Index');
ylabel('Measurement Value');

% Set initial radius for the circular object
circle_radius = 0.3;

% Function to update the circular object with the current radius
function update_plot(radius, img_1, stim, h1, homg_data)
    clear inhomg_data clear img_2
    % Create an image with the circular object
    img_2 = img_1;
    select_fcn = @(x, y, z) (x - 0.5).^2 + (y - 0.5).^2 < radius^2;
    img_2.elem_data = 1 + elem_select(img_2.fwd_model, select_fcn);
    img_2.fwd_model.stimulation = stim;
    
    % Display the FEM and measurement data
    subplot(2, 2, 3);
    show_fem(img_2);
    common_colourbar(h1, img_2);
    title('Inhomogeneous FEM with Circular Object');
    
    % Calculate and plot the difference in measurements
    
    inhomg_data = fwd_solve(img_2);
    % measurement_diff = homg_data.meas - inhomg_data.meas;
    measurement_diff = inhomg_data.meas;
    subplot(2, 2, 4);
    plot(1:length(measurement_diff), measurement_diff);
    title(radius);
    xlabel('Measurement Index');
    ylabel('Measurement Difference');
end
% Function to update the 2D Gaussian inhomogeneity using a select function
function update_plot_gaussian(sigma, img_1, stim, h1, homg_data)
    % Create an image with the Gaussian object
    img_2 = img_1;
    
    % Set the amplitude for the inhomogeneity (Gaussian value)
    A = 15;  % Conductivity value for the inhomogeneity
    B = 1;   % Background conductivity value (outside Gaussian region)
    % Define a select function for a 2D Gaussian
    sigma = 0.1;  % Standard deviation (controls spread of Gaussian)
    x0 = 0.5;     % Center of the Gaussian (x-coordinate)
    y0 = 0.625;   % Center of the Gaussian (y-coordinate)
    
    select_fcn = @(x, y, z) exp(-((x - x0).^2 + (y - y0).^2) / (2 * sigma^2));

    % Apply the select function to select elements inside the Gaussian
    img_2.elem_data = B + A * elem_select(img_2.fwd_model, select_fcn);
    
    % Set the stimulation pattern
    img_2.fwd_model.stimulation = stim;
    
    % Display the FEM with the Gaussian inhomogeneity
    subplot(2, 2, 3);
    show_fem(img_2);
    common_colourbar(h1, img_2);
    title('Inhomogeneous FEM with Gaussian Object');
    
    % Calculate and plot the difference in measurements
    inhomg_data = fwd_solve(img_2);
    measurement_diff = homg_data.meas - inhomg_data.meas;
    % measurement_diff = inhomg_data.meas;
    subplot(2, 2, 4);
    plot(1:length(measurement_diff), measurement_diff);
    ylim([-0.5, 0.5]);
    title('Difference: homg\_data.meas - inhomg\_data.meas');
    xlabel('Measurement Index');
    ylabel('Measurement Difference');
end




% Create a slider for adjusting the circle radius
slider = uicontrol('Style', 'slider', 'Min', 0, 'Max', 0.5, 'Value', circle_radius, ...
                   'Units', 'normalized', 'Position', [0.15 0.05 0.7 0.03], ...
                   'Callback', @(src, event) update_plot(get(src, 'Value'), img_1, stim, h1, homg_data));

% Initial plot
% update_plot(circle_radius, img_1, stim, h1, homg_data);
update_plot_gaussian(circle_radius, img_1, stim, h1, homg_data);
