% filepath: c:\Users\dhruv\Documents\Tripos\SoftTactileSensingForRoboticManipulation\Code\Analysis\TorqueFit.m

%% Global Configuration
global config; % Declare config as a global variable
config.centre = 1.8;
config.sigma = 0.3;
config.k = 5;

% Array of select_fcn options
config.select_fcns = {
    @(x, y, z) exp(-(y - config.centre).^2 / (2 * config.sigma^2)) .* cos(config.k * (y - config.centre)); % mod gauss
    @(x, y, z) -(y - config.centre) .* exp(-((y - config.centre).^2) / (2 * config.sigma^2)) / (config.sigma^2); % diff gauss
    @(x, y, z) 0.5 * (y > config.centre) - 0.5 * (y <= config.centre); % step
    @(x, y, z) (y - config.centre) / config.sigma; % linear
    @(x, y, z) (y > config.centre - config.k/2) .* (y < config.centre + config.k/2) .* (y - config.centre) / config.sigma; % linear with k
    @(x, y, z) (y <= config.centre) .* exp(-(y - config.centre).^2 / (2 * config.sigma^2)) + ...
               (y > config.centre) .* exp(-(y - config.centre).^2 / (2 * config.sigma^2)) .* cos(config.k * (y - config.centre)); % combined
    @(x, y, z) -(y - config.centre) .* exp(-((y - config.centre).^2) / (2 * config.sigma^2)) / (config.sigma^2); % DoG
    @(x, y, z) (abs(y - config.centre) < config.k/2) .* ...
               (0.5 * (1 + cos((2*pi/config.k) * (y - config.centre)))) .* ...
               (1 + config.sigma * sin(pi * (y - config.centre) / config.k)); % custom
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
    best_shift = find_best_shift_envelope(data_diff, sim_diff, 1, 896, 50);
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

function select_fcn = get_select_fcn()
    % Return the appropriate select function based on the global configuration
    global config;
    select_fcn = config.select_fcns{config.selected_fcn_idx};
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
    % Analyze correlation for different diameters
    base_dir = 'SavedVariables';
    diameters = {'10mm', '20mm', '30mm', '40mm'};
    k_const = 4;
    sigma_fixed = 4.333;
    k_values = linspace(0.8, 1.1, 20);
    sigma_values = linspace(2, 5, 10);
    smooth_coeff = 50;

    Corr_vs_K_AllDiameters(base_dir, diameters, sigma_fixed, k_values, smooth_coeff);
end

function best_shift = find_best_shift(data_diff, sim_diff, shift_step, max_shifts)
    biggest_cor = [0, 0]; % [correlation, shift index]
    corrs = zeros([max_shifts, 1]);

    for i = 1:max_shifts
        shifted_data = circshift(data_diff, i * shift_step);
        score = corr(shifted_data, sim_diff, 'Type', 'Spearman');
        corrs(i) = score;
        if score > biggest_cor(1)
            biggest_cor = [score, i];
        end
    end

    best_shift = biggest_cor(2) * shift_step;
    fprintf('Best shift: %d samples, Correlation: %.4f\n', best_shift, biggest_cor(1));
end

function best_shift = find_best_shift_dtw(data_diff, sim_diff, shift_step, max_shifts)
    min_dtw_dist = Inf; % Start with a very high DTW distance
    best_shift = 0;
    
    for i = 1:max_shifts
        shifted_data = circshift(data_diff, i * shift_step);
        dtw_dist = dtw(shifted_data, sim_diff); % Compute DTW distance
        if dtw_dist < min_dtw_dist
            min_dtw_dist = dtw_dist;
            best_shift = i * shift_step;
        end
    end

    fprintf('Best shift: %d samples (DTW distance: %.4f)\n', best_shift, min_dtw_dist);
end

function best_shift = find_best_shift_envelope(data_diff, sim_diff, shift_step, max_shifts, smooth_coeff)
    % Function to find the best shift using envelope correlation
    
    max_corr = -Inf;
    best_shift = 0;

    for i = 1:max_shifts
        shifted_data = circshift(data_diff, i * shift_step);
        score = envelope_correlation(shifted_data, sim_diff,smooth_coeff);  % Use the envelope correlation function

        if score > max_corr
            max_corr = score;
            best_shift = i * shift_step;
        end
    end

    fprintf('Best shift: %d samples (Envelope Correlation: %.4f)\n', best_shift, max_corr);
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


%% Function: Optimize k and sigma
function [best_k, best_sigma, best_corr] = optimize_parameters(data_diff, mdl,stim, k_values, sigma_values, smooth_coeff)
    best_corr = -Inf;
    best_k = 0;
    best_sigma = 0;
    best_env_1 = 0;
    best_env_2=0;

    for k = k_values
        for sigma = sigma_values
            % Generate new select_fcn using k and sigma
            centre = 1.8;
            % select_fcn = @(x,y,z) exp(-(y - centre).^2 / (2 * sigma^2)) .* (cos(k * (y - centre)));
            select_fcn = @(x, y, z) -(y - centre) .* exp(-((y - centre).^2) / (2 * sigma^2)) / (sigma^2);
            % Generate model
            plain = mk_image(mdl, 1, 'Hi');
            plain.fwd_model.stimulation = stim;
            press = plain;
            press.elem_data = 1 + elem_select(press.fwd_model, select_fcn);
            press.fwd_model.stimulation = stim;

            % Compute simulated difference
            plain_data = fwd_solve(plain);
            press_data = fwd_solve(press);
            sim_diff_test = abs(press_data.meas - plain_data.meas) / 10;

            % Compute envelope correlation
            [corr_score,env_1,env_2] = envelope_correlation(data_diff, sim_diff_test, smooth_coeff);

            % Check if this is the best correlation found
            if corr_score > best_corr
                best_corr = corr_score;
                best_k = k;
                best_sigma = sigma;
                best_env_1 = env_1;
                best_env_2= env_2;
            end
        end
    end
    fprintf('Best k: %.2f, Best sigma: %.2f, Best correlation: %.4f\n', best_k, best_sigma, best_corr);
    figure()
    hold on
    plot(best_env_1);plot(best_env_2);hold off;
    % Compute correlation
end

function [train_results, test_results] = find_params(data_dir, train_torques, test_torques, mdl, stim, k_values, sigma_values, smooth_coeff)
    % FIND_PARAMS: Computes optimized k and sigma for all training and test datasets
    
    % Get list of files
    train_files = dir(fullfile(data_dir, 'Train_*.mat'));
    test_files = dir(fullfile(data_dir, 'Test_*.mat'));

    % Ensure consistency
    if length(train_files) ~= length(train_torques) || length(test_files) ~= length(test_torques)
        error('Mismatch: Torque values do not match data files.');
    end

    % Process datasets
    train_results = process_files(train_files, train_torques, mdl, stim, k_values, sigma_values, smooth_coeff);
    test_results = process_files(test_files, test_torques, mdl, stim, k_values, sigma_values, smooth_coeff);
end

function results = process_files(file_list, torques, mdl, stim, k_values, sigma_values, smooth_coeff)
    % PROCESS_FILES: Helper function to process datasets and compute optimal k and sigma
    num_datasets = length(file_list);
    results = zeros(num_datasets, 3); % Columns: [Torque, K_opt, Sigma_opt]

    for i = 1:num_datasets
        % Load and transpose data
        file_path = fullfile(file_list(i).folder, file_list(i).name);
        data_diff = load(file_path).data_diff';  

        % Optimize k and sigma
        [best_k, best_sigma, ~] = optimize_parameters(data_diff, mdl, stim, k_values, sigma_values, smooth_coeff);

        % Store results
        results(i, :) = [torques(i), best_k, best_sigma];

        fprintf('Processed %s | Torque: %.2f | k_opt: %.2f | sigma_opt: %.2f\n', ...
            file_list(i).name, torques(i), best_k, best_sigma);
    end
end

function [k_fit, sigma_fit] = fit_torque(results)
    % FIT_TORQUE: Fits sinusoidal functions to training data
    
    torque_values = results(:,1);
    optimised_k = results(:,2);
    optimised_sigma = results(:,3);

    % Fit functions
    k_fit = fit(torque_values, optimised_k, 'poly3');
    sigma_fit = fit(torque_values, optimised_sigma, 'poly3');

    % Plot results
    figure;
    subplot(2,1,1);
    scatter(torque_values, optimised_k, 'bo'); hold on;
    plot(k_fit, 'b-');
    title('Optimized k vs Torque (Polynomial Fit)');
    xlabel('Torque');
    ylabel('k');
    
    subplot(2,1,2);
    scatter(torque_values, optimised_sigma, 'ro'); hold on;
    plot(sigma_fit, 'r-');
    title('Optimized sigma vs Torque (Polynomial Fit)');
    xlabel('Torque');
    ylabel('sigma');
end

function evaluate_fit(test_results, k_fit, sigma_fit)
    % EVALUATE_FIT: Compares predicted vs actual k and sigma for test data
    
    test_torques = test_results(:,1);
    actual_k = test_results(:,2);
    actual_sigma = test_results(:,3);

    % Predict values
    predicted_k = feval(k_fit, test_torques);
    predicted_sigma = feval(sigma_fit, test_torques);

    % Compute errors
    abs_error_k = abs(predicted_k - actual_k);
    abs_error_sigma = abs(predicted_sigma - actual_sigma);
    percent_error_k = 100 * abs_error_k ./ actual_k;
    percent_error_sigma = 100 * abs_error_sigma ./ actual_sigma;

    % RMSE
    rmse_k = sqrt(mean(abs_error_k.^2));
    rmse_sigma = sqrt(mean(abs_error_sigma.^2));
    fprintf('\nFit Performance:\nRMSE (k): %.4f, RMSE (sigma): %.4f\n', rmse_k, rmse_sigma);

    % Plot errors
    figure;
    subplot(2,1,1);
    scatter(test_torques, abs_error_k, 'bo'); hold on;
    scatter(test_torques, abs_error_sigma, 'ro');
    title('Torque vs. Absolute Error');
    xlabel('Torque');
    ylabel('Absolute Error');
    legend('|k_{error}|', '|sigma_{error}|');
    grid on;

    subplot(2,1,2);
    scatter(test_torques, percent_error_k, 'bo'); hold on;
    scatter(test_torques, percent_error_sigma, 'ro');
    title('Torque vs. Percentage Error');
    xlabel('Torque');
    ylabel('Percentage Error (%)');
    legend('% k_{error}', '% sigma_{error}');
    grid on;
end

function analyze_torque_fit(train_results, test_results)
    % ANALYZE_TORQUE_FIT: Plots optimized k and sigma for all torque values using binned torque values for coloring.
    %
    % Inputs:
    % - train_results: Matrix (N_train x 3) [Torque, k_opt, sigma_opt]
    % - test_results: Matrix (N_test x 3) [Torque, k_opt, sigma_opt]
    % - num_bins: Number of bins to group torque values (should be 10)

    % Combine train & test data
    all_results = [train_results; test_results];
    all_torques = all_results(:,1);
    all_k = all_results(:,2);
    all_sigma = all_results(:,3);

    % Create bins for torque values (forcing 10 bins)
    [~, ~, bin_idx] = histcounts(all_torques, 10);

    % Define a colormap with exactly 10 distinct colors
    cmap = lines(10);

    % Plot optimized k vs Torque
    figure;
    subplot(2,1,1); hold on;
    for i = 2:10
        idx = (bin_idx == i);
        scatter(all_torques(idx), all_k(idx), 50, cmap(i,:), 'filled', 'DisplayName', sprintf('Bin %d', i));
    end
    title('Optimized k vs Torque');
    xlabel('Torque');
    ylabel('k');
    legend('Location', 'best');
    grid on;

    % Plot optimized sigma vs Torque
    subplot(2,1,2); hold on;
    for i = 1:10
        idx = (bin_idx == i);
        scatter(all_torques(idx), all_sigma(idx), 50, cmap(i,:), 'filled', 'DisplayName', sprintf('Bin %d', i));
    end
    title('Optimized \sigma vs Torque');
    xlabel('Torque');
    ylabel('\sigma');
    legend('Location', 'best');
    grid on;
end
%% Correlation for diameters
function Corr_vs_Sigma_AllDiameters(base_dir, diameters, mdl, stim, k_fixed, sigma_values, smooth_coeff)
    % CORR_VS_SIGMA_ALLDIAMETERS: Computes and plots correlation vs sigma for multiple diameters
    
    % Define colors for different diameters
    colors = lines(length(diameters)); % Generate distinct colors
    optimal_sigmas = zeros(1, length(diameters)); % Store optimal sigma values

    figure; hold on;
    
    % Loop through each diameter
    for d = 1:length(diameters)
        diameter = diameters{d};
        data_dir = fullfile(base_dir, ['TorqueFitting_', diameter]);  % Set data path
        torque_file = fullfile(base_dir, 'TorqueFitting', ['Torque_', diameter, '.mat']);
        
        % Load torque values
        torque_data = load(torque_file);
        train_torque = torque_data.trainTorquePeaks';
        test_torque = torque_data.testTorquePeaks';
        torque_values = [train_torque, test_torque];

        % Get list of data files
        files = dir(fullfile(data_dir, '*.mat'));

        % Ensure torque values match file count
        if length(files) ~= length(torque_values)
            error('Mismatch: Number of torque values does not match number of data files for %s.', diameter);
        end

        % Initialize storage for results
        correlations = zeros(length(files), length(sigma_values));

        % Loop through each dataset
        for i = 1:length(files)
            % Load and transpose data
            file_path = fullfile(files(i).folder, files(i).name);
            data_diff = load(file_path).data_diff';
            % plot(envelope(data_diff,50,"peak"))
            % figure();
            % Loop through sigma values
            for j = 1:length(sigma_values)
                sigma = sigma_values(j);
                centre = 1.8;

                % select_fcn = @(x, y, z) exp(-(y - centre).^2 / (2 * sigma^2)) .* cos(k_fixed * (y - centre)); %mod gauss
                % select_fcn = @(x, y, z) -(y - centre) .* exp(-((y - centre).^2) / (2 * sigma^2)) / (sigma^2); %diff gauss
                % select_fcn = @(x, y, z) 0.5 * (y > centre) - 0.5 * (y <= centre); %step
                % select_fcn = @(x, y, z) (y - centre) / sigma; %linear
                % select_fcn = @(x,y,z)(y>centre-k_fixed/2).*(y<centre+k_fixed/2).*(y-centre)/sigma; %linear with k
                %select_fcn = @(x, y, z) (y <= centre) .* exp(-(y - centre).^2 / (2 * sigma^2)) + (y > centre) .* exp(-(y - centre).^2 / (2 * sigma^2)) .* cos(k_fixed * (y - centre));
                % select_fcn = @(x, y, z) -(y - centre) .* exp(-((y - centre).^2) / (2 * sigma^2)) / (sigma^2); %DoG
                select_fcn = @(x, y, z) ( (abs(y - centre) < k_fixed/2) ...
    .* (0.5 * (1 + cos((2*pi/k_fixed) * (y - centre)))) ...
    .* (1 + sigma * sin(pi*(y - centre)/k_fixed)) );

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
                correlations(i, j) = envelope_correlation(data_diff, sim_diff, smooth_coeff);
            end
        end
        
        % Compute average correlation for each sigma (across all torque values)
        avg_corr = mean(correlations, 1);
        smoothed_corr = smoothdata(avg_corr, SmoothingFactor=0);
        save(sprintf("Analysis\\Avg_Corr_%s.mat", diameter), 'avg_corr')

        % Plot results for this diameter
        plot(sigma_values, smoothed_corr, 'Color', colors(d, :), 'LineWidth', 2, ...
             'DisplayName', sprintf('%s Diameter', diameter));

        % Find the best sigma value
        [~, idx] = max(smoothed_corr);
        optimal_sigma = sigma_values(idx);
        optimal_sigmas(d) = optimal_sigma; % Store optimal sigma
        disp(optimal_sigma);
        xline(optimal_sigma, 'Color', colors(d, :), 'LineWidth', 2, 'DisplayName', sprintf('%s Diameter', diameter));
    end

    hold off;
    xlabel('Sigma');
    ylabel('Correlation');
    title(sprintf('Correlation vs Sigma for Fixed k = %.2f', k_fixed));
    legend('Location', 'best');
    grid on;

    % ---- PLOT ALL SELECT_FCN FUNCTIONS TOGETHER ----
    figure; hold on;
    y_range = linspace(centre - 3, centre + 3, 200); % Define range for y values

    for d = 1:length(diameters)
        % Define select_fcn with the optimal sigma found
        % select_fcn_opt = @(y) (y <= centre) .* exp(-(y - centre).^2 / (2 * optimal_sigmas(d)^2)) + (y > centre) .* exp(-(y - centre).^2 / (2 * optimal_sigmas(d)^2)) .* cos(k_fixed * (y - centre));
        % select_fcn_opt = @(y) -(y - centre) .* exp(-((y - centre).^2) / (2 * optimal_sigmas(d)^2)) / (optimal_sigmas(d)^2); %DoG
        select_fcn_opt= @( y) ( (abs(y - centre) < k_fixed/2) ...
    .* (0.5 * (1 + cos((2*pi/k_fixed) * (y - centre)))) ...
    .* (1 + sigma * sin(pi*(y - centre)/k_fixed)) );

        % Plot select_fcn with its optimal sigma
        plot(y_range, select_fcn_opt(y_range), 'Color', colors(d, :), 'LineWidth', 2, ...
             'DisplayName', sprintf('%s Diameter (\\sigma = %.3f)', diameters{d}, optimal_sigmas(d)));
    end

    xlabel('Position');
    ylabel('Select Function Value');
    title('Select Function for Different Diameters (Optimal \\sigma)');
    legend('Location', 'best');
    grid on;
    hold off;

    % ---- PLOT OPTIMAL ENVELOPE VS DATA_DIFF ENVELOPE ----
    % figure; hold on;
    % 
    % for d = 1:length(diameters)
    %     figure();
    %     % Reload a sample data file to compare envelopes
    %     file_path = fullfile(files(1).folder, files(1).name);
    %     data_diff = load(file_path).data_diff';
    % 
    %     % Compute the envelope of data_diff
    %     [env_data, ~] = envelope(data_diff, 50, 'peak');
    % 
    %     % Compute simulated envelope using the optimal sigma
    %     select_fcn_opt = @(x, y, z) -(y - centre) .* exp(-((y - centre).^2) / (2 * optimal_sigmas(d)^2)) / (optimal_sigmas(d)^2);
    %     plain = mk_image(mdl, 1, 'Hi');
    %     plain.fwd_model.stimulation = stim;
    %     press = plain;
    %     press.elem_data = 1 + elem_select(press.fwd_model, select_fcn_opt);
    %     press.fwd_model.stimulation = stim;
    % 
    %     % Compute simulated difference
    %     plain_data = fwd_solve(plain);
    %     press_data = fwd_solve(press);
    %     sim_diff = abs(press_data.meas - plain_data.meas) / 10;
    %     [env_sim, ~] = envelope(sim_diff, 50, 'peak');
    % 
    %     % Normalize envelopes for better shape comparison
    %     env_data = (env_data - min(env_data)) / (max(env_data) - min(env_data) + 1e-10);
    %     env_sim = (env_sim - min(env_sim)) / (max(env_sim) - min(env_sim) + 1e-10);
    % 
    %     % Plot envelopes
    %     plot(env_data, 'Color', colors(d, :), 'LineWidth', 2, 'LineStyle', '--', ...
    %          'DisplayName', sprintf('%s Diameter Data', diameters{d}));
    %     plot(env_sim, 'Color', colors(d, :), 'LineWidth', 2, ...
    %          'DisplayName', sprintf('%s Diameter Simulated', diameters{d}));
    % 
    %         xlabel('Sample Index');
    % ylabel('Normalized Envelope');
    % title('Comparison of Data and Simulated Envelopes');
    % legend('Location', 'best');
    % grid on;
    % hold off;
    % end


end


function Corr_vs_K_AllDiameters(base_dir, diameters, sigma_fixed, k_values, smooth_coeff)
    global config;
    % CORR_VS_K_ALLDIAMETERS: Computes and plots correlation vs k for multiple diameters (with fixed sigma)
    mdl=config.mdl;
    stim=config.stim;
    % Define colors for different diameters
    colors = lines(length(diameters)); % Generate distinct colors
    optimal_ks = zeros(1, length(diameters)); % Store optimal k values
    centre = 1.8;

    figure; hold on;

    for d = 1:length(diameters)
        diameter = diameters{d};
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

        correlations = zeros(length(files), length(k_values));

        for i = 1:length(files)
            file_path = fullfile(files(i).folder, files(i).name);
            data_diff = load(file_path).data_diff';

            for j = 1:length(k_values)
                k = k_values(j);

                select_fcn = @(x, y, z) ( (abs(y - centre) < k/2) ...
                    .* (0.5 * (1 + cos((2*pi/k) * (y - centre)))) ...
                    .* (1 + sigma_fixed * sin(pi*(y - centre)/k)) );

                plain = mk_image(mdl, 1, 'Hi');
                plain.fwd_model.stimulation = stim;
                press = plain;
                press.elem_data = 1 + elem_select(press.fwd_model, select_fcn);
                press.fwd_model.stimulation = stim;

                plain_data = fwd_solve(plain);
                press_data = fwd_solve(press);
                sim_diff = abs(press_data.meas - plain_data.meas) / 10;

                correlations(i, j) = envelope_correlation(data_diff, sim_diff, smooth_coeff);
            end
        end

        avg_corr = mean(correlations, 1);
        smoothed_corr = smoothdata(avg_corr, SmoothingFactor=0.5);
        save(sprintf("Analysis\\Avg_Corr_K_%s.mat", diameter), 'avg_corr')

        plot(k_values, smoothed_corr, 'Color', colors(d, :), 'LineWidth', 2, ...
             'DisplayName', sprintf('%s Diameter', diameter));

        [~, idx] = max(smoothed_corr);
        optimal_k = k_values(idx);
        optimal_ks(d) = optimal_k;
        disp(optimal_k);
        xline(optimal_k, 'Color', colors(d, :), 'LineWidth', 2, ...
              'DisplayName', sprintf('%s Diameter', diameter));
    end

    hold off;
    xlabel('k');
    ylabel('Correlation');
    title(sprintf('Correlation vs k for Fixed \\sigma = %.2f', sigma_fixed));
    legend('Location', 'best');
    grid on;

    % ---- PLOT ALL SELECT_FCN FUNCTIONS TOGETHER ----
    figure; hold on;
    y_range = linspace(centre - 3, centre + 3, 200);

    for d = 1:length(diameters)
        k = optimal_ks(d);
        select_fcn_opt = @(y) ( (abs(y - centre) < k/2) ...
            .* (0.5 * (1 + cos((2*pi/k) * (y - centre)))) ...
            .* (1 + sigma_fixed * sin(pi*(y - centre)/k)) );

        plot(y_range, select_fcn_opt(y_range), 'Color', colors(d, :), 'LineWidth', 2, ...
             'DisplayName', sprintf('%s Diameter (k = %.3f)', diameters{d}, k));
    end

    xlabel('Position');
    ylabel('Select Function Value');
    title('Select Function for Different Diameters (Optimal k)');
    legend('Location', 'best');
    grid on;
    hold off;
end



function Corr_vs_Sigma_K_AllDiameters(base_dir, diameters, mdl, stim, k_values, sigma_values, smooth_coeff)
    % CORR_VS_SIGMA_K_ALLDIAMETERS: Computes and plots correlation vs sigma & k for multiple diameters
    
    % Loop through each diameter
    for d = 1:length(diameters)
        diameter = diameters{d};
        data_dir = fullfile(base_dir, ['TorqueFitting_', diameter]);  % Set data path
        torque_file = fullfile(base_dir, 'TorqueFitting', ['Torque_', diameter, '.mat']);
        
        % Load torque values
        torque_data = load(torque_file);
        train_torque = torque_data.trainTorquePeaks';
        test_torque = torque_data.testTorquePeaks';
        torque_values = [train_torque, test_torque];
        
        % Get list of data files
        files = dir(fullfile(data_dir, '*.mat'));
        
        % Ensure torque values match file count
        if length(files) ~= length(torque_values)
            error('Mismatch: Number of torque values does not match number of data files for %s.', diameter);
        end
        
        % Initialize storage for results
        correlations = zeros(length(files), length(sigma_values), length(k_values));
        % Loop through each dataset
        for i = 1:length(files)
            % Load and transpose data
            file_path = fullfile(files(i).folder, files(i).name);
            data_diff = load(file_path).data_diff';
            
            % Loop through sigma and k values
            for j = 1:length(sigma_values)
                for k = 1:length(k_values)
                    sigma = sigma_values(j);
                    k_fixed = k_values(k);
                    centre = 1.8;
                    
                    % Select function
                    % select_fcn = @(x, y, z) exp(-(y - centre).^2 / (2 * sigma^2)) .* cos(k_fixed * (y - centre)); %mod gauss
                    % select_fcn = @(x, y, z) -(y - centre) .* exp(-((y - centre).^2) / (2 * sigma^2)) / (sigma^2); %diff gauss
                    % select_fcn = @(x, y, z) 0.5 * (y > centre) - 0.5 * (y <= centre); %step
                    % select_fcn = @(x, y, z) (y - centre) / sigma; %linear
                    %select_fcn = @(x,y,z)(y>centre-k_fixed/2).*(y<centre+k_fixed/2).*(y-centre)/sigma; %linear with k
                    %select_fcn = @(x, y, z) (y <= centre) .* exp(-(y - centre).^2 / (2 * sigma^2)) + (y > centre) .* exp(-(y - centre).^2 / (2 * sigma^2)) .* cos(k_fixed * (y - centre));
                    % select_fcn = @(x, y, z) -(y - centre) .* exp(-((y - centre).^2) / (2 * sigma^2)) / (sigma^2); %DoG
                    select_fcn = @(x, y, z) ( (abs(y - centre) < k/2) ...
    .* (0.5 * (1 + cos((2*pi/k) * (y - centre)))) ...
    .* (1 + sigma * sin(pi*(y - centre)/k)) );

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
                    correlations(i, j, k) = envelope_correlation(data_diff, sim_diff, smooth_coeff);
                end
            end
        end
        
        % Compute average correlation for each (sigma, k) pair (across all torque values)
        avg_corr = squeeze(mean(correlations, 1));
        smoothed_corr = smoothdata(avg_corr, SmoothingFactor=0);
        save(sprintf("Analysis\\Avg_Corr_K_%s.mat", diameter), 'avg_corr')

        % Create Heatmap

        figure;
        imagesc(k_values, sigma_values, smoothed_corr); % Create heatmap
        colorbar; % Display color scale
        xlabel('k Value');
        ylabel('Sigma');
        title(sprintf('Correlation vs Sigma & k for %s Diameter', diameter));
        axis xy; % Ensure correct orientation of the axes
        set(gca, 'YDir', 'normal'); % Y-axis in normal direction (small to large sigma)
        colormap hot; % Use jet colormap for better color visualization
        grid on;
    end
end

