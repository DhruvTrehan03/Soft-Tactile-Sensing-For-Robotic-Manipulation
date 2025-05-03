% filepath: c:\Users\dhruv\Documents\Tripos\SoftTactileSensingForRoboticManipulation\Code\Analysis\TorqueFit.m

%% Global Configuration
global config; % Declare config as a global variable
config.centre = 1.8;
config.sigma = 0.3;
config.k = 5;

% Array of select_fcn options
config.select_fcns = {
    @(x, y, z, k, sigma, centre) exp(-(y - centre).^2 / (2 * sigma^2)) .* cos(k * (y - centre)); % mod gauss
    @(x, y, z, k, sigma, centre) -(y - centre) .* exp(-((y - centre).^2) / (2 * sigma^2)) / (sigma^2); % diff gauss
    @(x, y, z, k, sigma, centre) 0.5 * (y > centre) - 0.5 * (y <= centre); % step
    @(x, y, z, k, sigma, centre) (y - centre) / sigma; % linear
    @(x, y, z, k, sigma, centre) (y > centre - k/2) .* (y < centre + k/2) .* (y - centre) / sigma; % linear with k
    @(x, y, z, k, sigma, centre) (y <= centre) .* exp(-(y - centre).^2 / (2 * sigma^2)) + ...
                                  (y > centre) .* exp(-(y - centre).^2 / (2 * sigma^2)) .* cos(k * (y - centre)); % combined
    @(x, y, z, k, sigma, centre) -(y - centre) .* exp(-((y - centre).^2) / (2 * sigma^2)) / (sigma^2); % DoG
    @(x, y, z, k, sigma, centre) (abs(y - centre) < k/2) .* ...
                                  (0.5 * (1 + cos((2*pi/k) * (y - centre)))) .* ...
                                  (1 + sigma * sin(pi * (y - centre) / k)); % custom
};

% Selected function index
config.selected_fcn_idx = 1; % Default to the first function

%% Main Script
% Toggle Settings
initialise = 0;
load_data = 1;
find_shift = 0;
test_mode = 0;
opt = 0;
params = 0;
fit = 0;
sigma_corr = 1;

% Initialize EIDORS
if initialise
    initialize_eidors();
end

% Load Data
if load_data
    [data_diff, sim_diff] = load_and_preprocess_data();
end

% Find Best Shift
if find_shift
    best_shift = find_best_shift(data_diff, sim_diff, 1, 896, 'envelope', 50);
    data_diff = circshift(data_diff, best_shift);
end

% Test Mode
if test_mode
    test_model(data_diff, sim_diff,config);
end

% Optimization
if opt optimize_parameters(data_diff); end

% Fit Torque to Model Parameters
if params fit_torque_to_model(); end

% Correlation Analysis
if sigma_corr analyze_correlation(); end

% Fit and Evaluate Torque
if fit fit_and_evaluate_torque(); end

%% Functions

function initialize_eidors()
    clear;
    global config;
    % Initialize EIDORS and define model parameters
    run('Source/eidors-v3.11-ng/eidors/eidors_startup.m');
    config.mdl = create_model();
    config.stim = create_stimulation(); % Use config.mdl
end

function mdl = create_model()
    % Create the FEM model
    height = 0;
    width = 4.4;
    len = 3.6;
    xy = [0 0 width width; 0 len len 0]';
    curve_type = 1;
    maxsz = 0.1;
    trunk_shape = {height, xy, curve_type, maxsz};
    elec_pos = [32, 1.1];
    elec_shape = 0.2;
    mdl = ng_mk_extruded_model(trunk_shape, elec_pos, elec_shape);
end

function stim = create_stimulation()
    % Create stimulation patterns
    stim = mk_stim_patterns(32, 1, [0, 16], [0, 1], {'no_meas_current'}, 5);
end

function [data_diff, sim_diff] = load_and_preprocess_data()
    % Load and Clean Electrode Data
    electrodeData = load("..\Readings\2024-12-05_18-15\device1.mat").Right_Data(1:6371,2:end);
    electrodeData = electrodeData(:, ~all(electrodeData == 0));
    trainingData = electrodeData(50:672, :);

    data_objs = load("SavedVariables\TorqueSlice.mat").clipped_data';
    data_homg = load("SavedVariables\TorqueSliceHom.mat").clipped_data_hom';
    data_objs = data_objs(data_objs ~= 0);
    data_homg = data_homg(data_homg ~= 0);
    data_diff = abs(data_objs - data_homg);

    % Apply Moving Average Filter
    windowSize = 10;
    b = (1 / windowSize) * ones(1, windowSize);
    data_diff = filter(b, 1, data_diff);

    % Adjust Data Alignment
    data_diff = circshift(data_diff, 615);

    % Placeholder for sim_diff (to be computed later)
    sim_diff = [];
end

function test_model(data_diff)
    % Test model and compare simulation
    select_fcn = get_select_fcn();
    press = create_press_model(config.mdl, select_fcn);
    plain_data = fwd_solve(config.mdl);
    press_data = fwd_solve(press);
    sim_diff = abs(press_data.meas - plain_data.meas);

    % Compute and plot correlation
    [correlation, env_data_diff_smooth, env_sim_diff_smooth] = envelope_correlation(data_diff, sim_diff, 50);
    plot_results(env_data_diff_smooth, env_sim_diff_smooth, press);
end



function select_fcn = get_select_fcn(template, k, sigma, centre)
    select_fcn = @(x, y, z) template(x, y, z, k, sigma, centre);
end

function press = create_press_model(mdl, select_fcn)
    % Create a press model with the given select function
    press = mk_image(mdl, 1, 'Hi');
    press.elem_data = 1 + elem_select(press.fwd_model, select_fcn);
    press.fwd_model.stimulation = mdl.fwd_model.stimulation;
end

function plot_results(env_data_diff_smooth, env_sim_diff_smooth, press)
    % Plot results
    figure;
    subplot(3, 1, 1);
    show_fem(press);
    subplot(3, 1, 2);
    plot(env_data_diff_smooth, 'r', 'LineWidth', 1.5);
    title('Data Difference with Envelope');
    subplot(3, 1, 3);
    plot(env_sim_diff_smooth, 'r', 'LineWidth', 1.5);
    title('Simulation Difference with Envelope');
end

function analyze_correlation()
    % ANALYZE_CORRELATION: Main function to analyze correlation for different diameters.
    global config;

    % Base directory and parameters
    base_dir = 'SavedVariables';
    diameters = {'10mm', '20mm', '30mm', '40mm'};
    param_ranges.k = linspace(0.8, 1.1, 20);
    param_ranges.sigma = linspace(2, 5, 10);
    smooth_coeff = 50;

    % Analyze correlation for all diameters
    analyze_diameters(base_dir, diameters, param_ranges, false, true, smooth_coeff);
end

function analyze_diameters(base_dir, diameters, param_ranges, fit_flag, plot_flag, smooth_coeff)
    % ANALYZE_DIAMETERS: Analyze correlation for multiple diameters and plot optimum k and sigma.
    global config;
    mdl = config.mdl;
    stim = config.stim;

    % Initialize storage for optimum k and sigma values
    optimal_k = zeros(1, length(diameters));
    optimal_sigma = zeros(1, length(diameters));

    % Loop through each diameter
    for d = 1:length(diameters)
        diameter = diameters{d};
        fprintf('Processing diameter: %s\n', diameter);

        % Load data
        data_dir = fullfile(base_dir, ['TorqueFitting_', diameter]);
        torque_file = fullfile(base_dir, 'TorqueFitting', ['Torque_', diameter, '.mat']);
        torque_data = load(torque_file);
        train_torque = torque_data.trainTorquePeaks';
        test_torque = torque_data.testTorquePeaks';
        torque_values = [train_torque, test_torque];

        files = dir(fullfile(data_dir, '*.mat'));
        if length(files) ~= length(torque_values)
            error('Mismatch: Number of torque values does not match number of data files for %s.', diameter);
        end

        % Initialize storage for correlations
        correlations = zeros(length(files), length(param_ranges.sigma), length(param_ranges.k));

        % Loop through each dataset
        for i = 1:length(files)
            file_path = fullfile(files(i).folder, files(i).name);
            data_diff = load(file_path).data_diff';

            % Compute correlations for all parameter combinations
            correlations(i, :, :) = compute_correlations(data_diff, mdl, stim, param_ranges, smooth_coeff);
        end

        % Compute average correlation
        avg_corr = squeeze(mean(correlations, 1));
        save(sprintf("Analysis\\Avg_Corr_%s.mat", diameter), 'avg_corr');

        % Find the optimum k and sigma values
        [max_corr, max_idx] = max(avg_corr(:));
        [opt_sigma_idx, opt_k_idx] = ind2sub(size(avg_corr), max_idx);
        optimal_k(d) = param_ranges.k(opt_k_idx);
        optimal_sigma(d) = param_ranges.sigma(opt_sigma_idx);

        % Plot heatmap for this diameter (if requested)
        if plot_flag
            plot_correlation_heatmap(avg_corr, param_ranges, diameter);
        end
    end

    % Plot optimum k and sigma values for all diameters
    figure;
    subplot(2, 1, 1);
    plot(1:length(diameters), optimal_k, '-o', 'LineWidth', 2);
    xticks(1:length(diameters));
    xticklabels(diameters);
    xlabel('Diameter');
    ylabel('Optimum k');
    title('Optimum k for Different Diameters');
    grid on;

    subplot(2, 1, 2);
    plot(1:length(diameters), optimal_sigma, '-o', 'LineWidth', 2);
    xticks(1:length(diameters));
    xticklabels(diameters);
    xlabel('Diameter');
    ylabel('Optimum \sigma');
    title('Optimum \sigma for Different Diameters');
    grid on;
end

function compute_correlations(data_diff, mdl, stim, param_ranges, smooth_coeff)
    % COMPUTE_CORRELATIONS: Compute correlation matrix for given data and parameter ranges.
    global config;
    select_fcn_template = config.select_fcns{config.selected_fcn_idx};
    centre = config.centre;

    % Initialize correlation matrix
    correlation_matrix = zeros(length(param_ranges.sigma), length(param_ranges.k));

    % Loop through sigma and k values
    for j = 1:length(param_ranges.sigma)
        for k = 1:length(param_ranges.k)
            sigma = param_ranges.sigma(j);
            k_fixed = param_ranges.k(k);

            % Generate select function
            select_fcn = @(x, y, z) select_fcn_template(x, y, z, k_fixed, sigma, centre);

            % Generate model
            plain = mk_image(mdl, 1, 'Hi');
            plain.fwd_model.stimulation = stim;
            press = plain;
            press.elem_data = 1 + elem_select(press.fwd_model, select_fcn);
            press.fwd_model.stimulation = stim;

            % Compute simulated difference
            plain_data = fwd_solve(plain);
            press_data = fwd_solve(press);
            sim_diff = abs(press_data.meas - plain_data.meas) / 10;

            % Compute correlation
            correlation_matrix(j, k) = envelope_correlation(data_diff, sim_diff, smooth_coeff);
        end
    end
end

function plot_correlation_heatmap(avg_corr, param_ranges, diameter)
    % PLOT_CORRELATION_HEATMAP: Plot heatmap of correlation vs sigma and k.
    figure;
    imagesc(param_ranges.k, param_ranges.sigma, avg_corr);
    colorbar;
    xlabel('k');
    ylabel('Sigma');
    title(sprintf('Correlation vs k & Sigma for %s Diameter', diameter));
    axis xy;
    colormap hot;
    grid on;
end

function best_shift = find_best_shift(data_diff, sim_diff, shift_step, max_shifts, shift_type, smooth_coeff)
    % FIND_BEST_SHIFT: Finds the best shift based on the specified shift type.
    %
    % Inputs:
    % - data_diff: The data difference array.
    % - sim_diff: The simulation difference array.
    % - shift_step: Step size for shifting.
    % - max_shifts: Maximum number of shifts to evaluate.
    % - shift_type: Type of shift ('envelope', 'dtw', 'correlation').
    % - smooth_coeff: Smoothing coefficient (only used for 'envelope').
    %
    % Output:
    % - best_shift: The best shift value.

    if nargin < 5
        shift_type = 'envelope'; % Default to envelope correlation
    end

    max_corr = -Inf; % Initialize for envelope and correlation
    min_dtw_dist = Inf; % Initialize for DTW
    best_shift = 0;

    for i = 1:max_shifts
        shifted_data = circshift(data_diff, i * shift_step);

        switch lower(shift_type)
            case 'envelope'
                % Use envelope correlation
                score = envelope_correlation(shifted_data, sim_diff, smooth_coeff);

                if score > max_corr
                    max_corr = score;
                    best_shift = i * shift_step;
                end

            case 'dtw'
                % Use Dynamic Time Warping (DTW) distance
                dtw_dist = dtw(shifted_data, sim_diff);

                if dtw_dist < min_dtw_dist
                    min_dtw_dist = dtw_dist;
                    best_shift = i * shift_step;
                end

            case 'correlation'
                % Use Spearman correlation
                score = corr(shifted_data, sim_diff, 'Type', 'Spearman');

                if score > max_corr
                    max_corr = score;
                    best_shift = i * shift_step;
                end

            otherwise
                error('Invalid shift_type: %s. Choose "envelope", "dtw", or "correlation".', shift_type);
        end
    end

    % Print the result
    switch lower(shift_type)
        case 'envelope'
            fprintf('Best shift: %d samples (Envelope Correlation: %.4f)\n', best_shift, max_corr);
        case 'dtw'
            fprintf('Best shift: %d samples (DTW distance: %.4f)\n', best_shift, min_dtw_dist);
        case 'correlation'
            fprintf('Best shift: %d samples (Spearman Correlation: %.4f)\n', best_shift, max_corr);
    end
end

function [corr_score,env1,env2] = envelope_correlation(data1, data2,smooth_coeff)
    % Extend the signals to reduce edge effects
    data1_ext = [data1; data1; data1];  % Triplicate the data
    data2_ext = [data2; data2; data2];

    % Compute the envelope of the extended signals
    [env1_ext, ~] = envelope(data1_ext, 50, 'peak');
    [env2_ext, ~] = envelope(data2_ext, 50, 'peak');

    % Extract only the middle section to avoid edge artifacts
    N = length(data1);
    env1 = env1_ext(N+1:2*N);
    env2 = env2_ext(N+1:2*N);

    % Smooth envelopes
    windowSize = 5;
    b = (1/windowSize) * ones(1, windowSize);
    a = 1;
    env1 = filter(b, a, env1);
    env2 = filter(b, a, env2);

    env1 = (env1 - min(env1)) / (max(env1) - min(env1));
    env2 = (env2 - min(env2)) / (max(env2) - min(env2));

    % Compute correlation
    corr_score = corr(env1, env2, 'Type', 'Spearman');  
    % corr_score = dtw(env1, env2);
    % corr_score = mean((env1 - env2).^2);  % Lower MSE = more simila
end

function [best_params, best_corr, fit_models] = optimize_and_fit(select_fcn_template, param_ranges, data_diff, mdl, stim, torque_values, smooth_coeff, fit)
    best_corr = -Inf;
    best_params = struct();
    fit_models = struct();

    % Generate all combinations of parameter values
    [param_grids{1:numel(fieldnames(param_ranges))}] = ndgrid(param_ranges.k, param_ranges.sigma);
    param_combinations = cell2mat(cellfun(@(x) x(:), param_grids, 'UniformOutput', false));

    % Loop through all parameter combinations
    for i = 1:size(param_combinations, 1)
        k = param_combinations(i, 1);
        sigma = param_combinations(i, 2);
        centre = 1.8; % Or pass this as a parameter if needed

        % Generate select_fcn using the current parameters
        select_fcn = @(x, y, z) select_fcn_template(x, y, z, k, sigma, centre);

        % Generate model
        plain = mk_image(mdl, 1, 'Hi');
        plain.fwd_model.stimulation = stim;
        press = plain;
        press.elem_data = 1 + elem_select(press.fwd_model, select_fcn);
        press.fwd_model.stimulation = stim;

        % Compute simulated difference
        plain_data = fwd_solve(plain);
        press_data = fwd_solve(press);
        sim_diff = abs(press_data.meas - plain_data.meas) / 10;

        % Compute envelope correlation
        corr_score = envelope_correlation(data_diff, sim_diff, smooth_coeff);

        % Update best parameters if correlation improves
        if corr_score > best_corr
            best_corr = corr_score;
            best_params.k = k;
            best_params.sigma = sigma;
        end
    end

    % Fit parameters to torque values (if applicable)
    if fit && !isempty(torque_values)
        fit_models.k = fit(torque_values, best_params.k, 'poly3');
        fit_models.sigma = fit(torque_values, best_params.sigma, 'poly3');
    end
end
