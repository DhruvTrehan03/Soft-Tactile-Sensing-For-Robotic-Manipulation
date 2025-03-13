%% Preamble

% Toggle Settings
initialise = false;
load_data = false;       
find_shift = false;      
test_mode = false;
opt = false;
fit = true;

% Initialize EIDORS (Only required the first time in a session)
if initialise
    clear;
    load_data = false;
    find_shift = false;
    test_mode = false;
    opt = false;
    fit = false;
    smooth_coeff = 50;
    
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
end

%% Load Data (Only If Needed)
if load_data
    % Load and Clean Electrode Data
    electrodeData = load("..\Readings\2024-12-05_18-15\device1.mat").Right_Data(1:6371,2:end);
    electrodeData = electrodeData(:, ~all(electrodeData == 0));
    trainingData = electrodeData(50:672, :);
    
    % Load Torque Data
    data_objs = load("SavedVariables\TorqueSlice.mat").clipped_data';
    data_homg = load("SavedVariables\TorqueSliceHom.mat").clipped_data_hom';
    
    % Remove Zero Values
    data_objs = data_objs(data_objs ~= 0);  
    data_homg = data_homg(data_homg ~= 0);  
    data_diff = abs(data_objs - data_homg);
    
    % Apply Moving Average Filter
    windowSize = 10;
    b = (1/windowSize) * ones(1, windowSize);
    data_diff = filter(b, 1, data_diff);
    
    % Adjust Data Alignment
    data_diff = circshift(data_diff, 615);
end

%% Find Best Shift (If Enabled)
if find_shift
    best_shift = find_best_shift_envelope(data_diff, sim_diff, 1, 896, smooth_coeff);
    data_diff = circshift(data_diff, best_shift);
end

%% Test Mode: Generate Models and Compare Simulation
if test_mode
    % Define Function for Model Selection
    centre = 1.8;
    sigma = 0.3;
    k = 5;
    select_fcn = @(x, y, z) exp(-(y - centre).^2 / (2 * sigma^2)) .* cos(k * (y - centre));
    
    % Generate and Solve Model
    press = plain;
    press.elem_data = 1 + elem_select(press.fwd_model, select_fcn);
    press.fwd_model.stimulation = stim;
    
    plain_data = fwd_solve(plain);
    press_data = fwd_solve(press);
    
    % Compute Difference
    sim_diff = abs(press_data.meas - plain_data.meas) / 10;
    [correlation, env_data_diff_smooth, env_sim_diff_smooth] = envelope_correlation(data_diff, sim_diff, smooth_coeff);
    
    % Plot Results
    figure;
    subplot(3,1,1);
    show_fem(press);
    
    subplot(3,1,2);
    hold on;
    plot(env_data_diff_smooth, 'r', 'LineWidth', 1.5);
    hold off;
    title('Data Difference with Envelope');
    legend('Envelope');
    
    subplot(3,1,3);
    hold on;
    plot(env_sim_diff_smooth, 'r', 'LineWidth', 1.5);
    hold off;
    title('Simulation Difference with Envelope');
    legend('Envelope');
end

%% Optimization
if opt
    % Define Parameter Ranges
    k_values = linspace(1, 10, 10);
    sigma_values = linspace(0.1, 1, 10);
    
    % Optimize k and sigma
    [best_k, best_sigma, best_corr] = optimize_parameters(data_diff, mdl, stim, k_values, sigma_values, smooth_coeff);
end

%% Fit Torque to Model Parameters
if fit
    data_dir = "SavedVariables\TorqueFitting";
    train_torque = load("SavedVariables\TorqueFitting\Torque.mat").trainTorquePeaks';
    
    k_values = linspace(1, 20, 20);
    sigma_values = linspace(0, 1, 20);
    smooth_coeff = 50;
    
    fit_torque_to_params(data_dir, train_torque, mdl, stim, k_values, sigma_values, smooth_coeff);
end


%% Functions

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
    % Compute the envelope of both signals
    [env1, ~] = envelope(data1, 10, 'peak');
    [env2, ~] = envelope(data2, 10, 'peak');

    % Smooth envelopes
    windowSize = smooth_coeff;
    b = (1/windowSize) * ones(1, windowSize);
    a = 1;
    env1 = filter(b, a, env1);
    env2 = filter(b, a, env2);

    % Compute correlation
    corr_score = corr(env1, env2, 'Type', 'Spearman');  % Spearman correlation for trend matching
end


%% Function: Optimize k and sigma
function [best_k, best_sigma, best_corr] = optimize_parameters(data_diff, mdl,stim, k_values, sigma_values, smooth_coeff)
    best_corr = -Inf;
    best_k = 0;
    best_sigma = 0;

    for k = k_values
        for sigma = sigma_values
            % Generate new select_fcn using k and sigma
            centre = 1.8;
            select_fcn = @(x,y,z) exp(-(y - centre).^2 / (2 * sigma^2)) .* (cos(k * (y - centre)));

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
            corr_score = envelope_correlation(data_diff, sim_diff_test, smooth_coeff);

            % Check if this is the best correlation found
            if corr_score > best_corr
                best_corr = corr_score;
                best_k = k;
                best_sigma = sigma;
            end
        end
    end
    fprintf('Best k: %.2f, Best sigma: %.2f, Best correlation: %.4f\n', best_k, best_sigma, best_corr);
end

function fit_torque_to_params(data_dir, torque_values, mdl, stim, k_values, sigma_values, smooth_coeff)
    % FIT_TORQUE_TO_PARAMS Optimizes k and sigma for different datasets and fits functions using sinusoids
    %
    % Inputs:
    % - data_dir: String, directory where the Train_*.mat files are stored
    % - torque_values: Array of corresponding torque values
    % - mdl: EIDORS model
    % - stim: Stimulation pattern
    % - k_values: Range of k values to optimize
    % - sigma_values: Range of sigma values to optimize
    % - smooth_coeff: Smoothing coefficient for envelope correlation

    % Get list of all "Train_*.mat" files in the directory
    file_list = dir(fullfile(data_dir, 'Train_*.mat'));
    num_datasets = length(file_list);

    if num_datasets ~= length(torque_values)
        error('Number of torque values must match number of Train_*.mat files.');
    end

    % Initialize arrays to store optimized values
    optimized_k = zeros(1, num_datasets);
    optimized_sigma = zeros(1, num_datasets);

    % Loop through each dataset
    for i = 1:num_datasets
        % Load precomputed data difference
        file_path = fullfile(data_dir, file_list(i).name);
        data_diff = load(file_path).data_diff';

        % Optimize parameters
        [best_k, best_sigma, ~] = optimize_parameters(data_diff, mdl, stim, k_values, sigma_values, smooth_coeff);

        % Store results
        optimized_k(i) = best_k;
        optimized_sigma(i) = best_sigma;

        fprintf('Dataset: %s | Best k: %.2f | Best sigma: %.2f\n', file_list(i).name, best_k, best_sigma);
    end

    % Fit sinusoidal functions
    ft = fittype('a0 + a1*sin(w*t) + b1*cos(w*t) + a2*sin(2*w*t) + b2*cos(2*w*t)', ...
                 'independent', 't', 'coefficients', {'a0', 'a1', 'b1', 'a2', 'b2', 'w'});

    k_fit = fit(torque_values', optimized_k', 'poly5', 'StartPoint', [mean(optimized_k), 1, 1, 1, 1, 0.1]);
    sigma_fit = fit(torque_values', optimized_sigma', ft, 'StartPoint', [mean(optimized_sigma), 1, 1, 1, 1, 0.1]);

    % Plot results
    figure;
    subplot(2,1,1);
    scatter(torque_values, optimized_k, 'bo'); hold on;
    plot(k_fit, 'b-');
    title('Optimized k vs Torque (Sinusoidal Fit)');
    xlabel('Torque');
    ylabel('k');
    
    subplot(2,1,2);
    scatter(torque_values, optimized_sigma, 'ro'); hold on;
    plot(sigma_fit, 'r-');
    title('Optimized sigma vs Torque (Sinusoidal Fit)');
    xlabel('Torque');
    ylabel('sigma');

    % Output fitted functions
    fprintf('Fitted k function: %s\n', formula(k_fit));
    fprintf('Fitted sigma function: %s\n', formula(sigma_fit));
end

