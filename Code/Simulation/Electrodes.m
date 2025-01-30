% Main Script
run('Source/eidors-v3.11-ng/eidors/eidors_startup.m'); % Initialize EIDORS

% Parameters that define the grid and electrodes
grid_params = struct('x_min', 0, 'x_max', 1, 'y_min', 0, 'y_max', 1.25, ...
                     'x_points', 80, 'y_points', 100, 'x_elec', 8, 'y_elec', 10);
elec_shape = 0.4;

% Generate Grid and Electrode Layout
[xy, exy] = generate_layout(grid_params);

% Create FEM Model
mdl = ng_mk_2d_model(xy, exy);

% Select function type ('circle', 'gaussian', 'sigmoid') and its parameters
select_fcn_type = 'gaussian'; % Options: 'circle', 'gaussian', 'sigmoid'
params = struct('center', [0.5, 0.625], 'radius', 0.0001, 'sigma', 0.2); % Adjust as needed

% Show FEM Model and Results
stim = mk_stim_patterns(32, 1, '{op}', '{ad}', {'no_meas_current', 'no_redundant'}, 1);

% First Inhomogeneity Plot
img = mk_image(mdl, 5); % Background conductivity is 5
select_fcn = get_select_fcn(select_fcn_type, params); % Get the selection function
img.elem_data = 5 + 10 * elem_select(img.fwd_model, select_fcn); % Apply the inhomogeneity

figure;
subplot(1, 2, 1);
show_fem(img);
view(90, 90); % Rotate plot 90 degrees
title(sprintf('FEM for a point touch (Gauss)'));
set(gca, 'XTick', [], 'YTick', []);

% Second Inhomogeneity Plot
select_fcn_type = 'diff\_gaussian\_1d'; % Change to differential Gaussian
img = mk_image(mdl, 5); % Background conductivity is 5
select_fcn = get_select_fcn(select_fcn_type, params); % Get the new selection function
img.elem_data = 5 + 10 * elem_select(img.fwd_model, select_fcn); % Apply the new inhomogeneity

subplot(1, 2, 2);
show_fem(img);
view(90, 90); % Rotate plot 90 degrees
title(sprintf('FEM with a torque (Diff Gauss)'));
set(gca, 'XTick', [], 'YTick', []);

% Add and label the colorbar below the plots
cb = colorbar('southoutside'); % Place colorbar below the plots
ylabel(cb, 'Conductivity (S/m)', 'FontSize', 12, 'FontWeight', 'bold'); % Label the colorbar

% Adjust the colorbar position and width if needed
cb.Position = [0.25, 0.16, 0.5, 0.03]; % [left, bottom, width, height]
fontsize(24,"points")

disp('Done.');
%% Function Definitions

function [xy, exy] = generate_layout(params)
    % Generate the grid layout (xy) and electrode layout (exy)

    % Generate the x-coordinates (horizontal lines)
    x1 = linspace(params.x_min, params.x_max, params.x_points)'; % Bottom
    x2 = linspace(params.x_max, params.x_min, params.x_points)'; % Top


    % Generate the y-coordinates (vertical lines)
    y1 = params.y_min * ones(size(x1)); % Bottom
    y2 = linspace(params.y_min, params.y_max, params.y_points)'; % Right
    y3 = params.y_max * ones(size(x2)); % Top
    y4 = linspace(params.y_max, params.y_min, params.y_points)'; % Left

    % Full grid layout (xy)
    xy = [
        [x1, y1]; % Bottom
        [params.x_max * ones(size(y2(2:end))), y2(2:end)]; % Right
        [x2(2:end), y3(2:end)]; % Top
        [params.x_min * ones(size(y4(2:end-1))), y4(2:end-1)]; % Left
    ];

    % Generate electrode positions
    ex1 = linspace(params.x_min, params.x_max, params.x_elec)'; % Bottom
    ex2 = linspace(params.x_max, params.x_min, params.x_elec)'; % Top
    ey1 = params.y_min * ones(size(ex1)); % Bottom
    ey2 = linspace(params.y_min, params.y_max, params.y_elec)'; % Right
    ey3 = params.y_max * ones(size(ex2)); % Top
    ey4 = linspace(params.y_max, params.y_min, params.y_elec)'; % Left

    % Combine electrode positions (exy)
    exy = [
        [ex1, ey1]; % Bottom
        [params.x_max * ones(size(ey2(2:end))), ey2(2:end)]; % Right
        [ex2(2:end), ey3(2:end)]; % Top
        [params.x_min * ones(size(ey4(2:end-1))), ey4(2:end-1)]; % Left
    ];
end

function plot_fem_results(mdl, stim, select_fcn_type, params)
    % Plot FEM results with and without added inhomogeneity

    % Subplot 1: Show FEM with homogeneous conductivity
    figure;
    img_1 = mk_image(mdl, 5);
    img_1.fwd_model.stimulation = stim;
    subplot(2, 2, 1);
    show_fem(img_1);
    title("FEM No Press");


    % Subplot 2: Homogeneous data plot
    homg_data = fwd_solve(img_1);
    subplot(2, 2, 2);
    plot(1:length(homg_data.meas), homg_data.meas);
    title("Electrode Pairing vs. Magnitude (Homogeneous)");

    % Subplot 3: Add an inhomogeneity
    img_2 = img_1;
    select_fcn = get_select_fcn(select_fcn_type, params);
    img_2.elem_data = 5 + 10 * elem_select(img_2.fwd_model, select_fcn);
    img_2.fwd_model.stimulation = stim;
    subplot(2, 2, 3);
    show_fem(img_2);
    title("FEM Press");
    common_colourbar(subplot(2, 2, 1), img_2);

    % Subplot 4: Inhomogeneous data plot
    inhomg_data = fwd_solve(img_2);
    subplot(2, 2, 4);
    plot(1:length(inhomg_data.meas), inhomg_data.meas);
    title("Electrode Pairing vs. Magnitude (Press)");

    % Figure 2: Difference in magnitude
    figure;
    plot(1:length(homg_data.meas), homg_data.meas - inhomg_data.meas);
    title("Difference in Magnitude");
end

function select_fcn = get_select_fcn(type, params)
    % Get the selection function based on type and parameters
    switch type
        case 'circle'
            center = params.center;
            radius = params.radius;
            select_fcn = @(x, y, z) (x - center(1)).^2 + (y - center(2)).^2 < radius^2;

        case 'gaussian'
            center = params.center;
            sigma = params.sigma;
            select_fcn = @(x, y, z) exp(-((x - center(1)).^2 + (y - center(2)).^2) / (2 * sigma^2));

        case 'sigmoid'
            center = params.center;
            scale = params.sigma; % Scaling factor for transition sharpness
            select_fcn = @(x, y, z) 1 ./ (1 + exp(-scale * (x - center(1)))) - 0.5; % Centered at 0
        case 'diff\_gaussian\_1d'
            % Differential of a 1D Gaussian along x-axis, constant across y
            center = params.center; % [center_x, center_y]
            sigma = params.sigma;   % Standard deviation of the Gaussian
            select_fcn = @(x, y, z) -(x - center(1)) .* exp(-((x - center(1)).^2) / (2 * sigma^2)) / (sigma^2);
        otherwise
            error('Unknown select_fcn_type: %s', type);
    end
end

function plot_fem_with_inhomogeneity(mdl, select_fcn_type, params)
    % Plot the FEM model with different inhomogeneities (circle, gaussian, sigmoid)
    %
    % Inputs:
    %   mdl             - FEM model
    %   select_fcn_type - Type of selection function ('circle', 'gaussian', 'sigmoid')
    %   params          - Parameters for the selection function

    % Create a homogeneous image
    img = mk_image(mdl, 5); % Background conductivity is 5

    % Get the selection function
    select_fcn = get_select_fcn(select_fcn_type, params);

    % Apply the inhomogeneity to the FEM model
    img.elem_data = 5 + 10 * elem_select(img.fwd_model, select_fcn);

    % Plot the FEM with the inhomogeneity
    figure;
    show_fem(img);
    title(sprintf('FEM with %s Inhomogeneity', select_fcn_type));

    % Add and label the colorbar
    cb = colorbar;
    ylabel(cb, 'Conductivity (S/m)', 'FontSize', 12, 'FontWeight', 'bold'); % Adjust label text as needed

    % Adjust the colorbar position if necessary
    set(cb, 'Position', [0.85, 0.2, 0.03, 0.6]); % [left, bottom, width, height]
end
