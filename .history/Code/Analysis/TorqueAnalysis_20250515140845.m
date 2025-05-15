clear; close all; clc;

% Define parameters
base_dir = 'SavedVariables\Torque_Slices';
TFit(base_dir);
base_dir = 'SavedVariables\Torque_Slices_Grouped';
TFit(base_dir);

function TFit(base_dir)
    % Load all data slice files
    files = dir(fullfile(base_dir, 'Data_Slice_*.mat'));
    torque_values = zeros(1, length(files));
    maxima_k = zeros(1, length(files));
    maxima_sigma = zeros(1, length(files));

    % Load the precomputed matrix for varying k and sigma
    load('sim_data_matrix.mat', 'sim_data_matrix', 'k_values', 'sigma_values');
    waitbar_handle = waitbar(0, 'Processing Data Slices...');
    heatmaps = zeros(length(k_values), length(sigma_values), length(files)); % Store heatmaps for each slice

    % Loop through each data slice
    for i = 1:length(files)
        % Update progress bar
        waitbar(i / length(files), waitbar_handle, sprintf('Processing Data Slice %d of %d...', i, length(files)));
        % Load data slice and torque value
        file_path = fullfile(files(i).folder, files(i).name);
        data = load(file_path);
        % data_diff = smoothdata(data.data_diff', "gaussian", 0); % Smooth data_diff
        data_diff = data.data_diff;
        torque_values(i) = data.torque_value;

        % Initialize heatmap accumulator
        heatmap_accumulator = zeros(length(k_values), length(sigma_values));

        % Compute cross-correlation for each k and sigma
        for k_idx = 1:length(k_values)
            for sigma_idx = 1:length(sigma_values)
                sim_data = squeeze(sim_data_matrix(k_idx, sigma_idx, :));
                [cross_corr, ~] = xcorr(data_diff, sim_data, 'coeff');
                heatmap_accumulator(k_idx, sigma_idx) = max(cross_corr); % Accumulate maximum correlation
            end
        end

        % Store the heatmap for the current data slice
        heatmaps(:, :, i) = heatmap_accumulator;

        % Find the global maxima
        [max_val, max_idx] = max(heatmap_accumulator, [], 'all', 'linear');
        [max_k_idx, max_sigma_idx] = ind2sub(size(heatmap_accumulator), max_idx);

        % Store the maxima for final plot
        maxima_k(i) = k_values(max_k_idx);
        maxima_sigma(i) = sigma_values(max_sigma_idx);

        % Generate heatmap for the current data slice
        % figure;
        % imagesc(sigma_values, k_values, heatmap_accumulator);
        % colorbar;
        % xlabel('Sigma');
        % ylabel('k');
        % title(sprintf('Cross-Correlation Heatmap for Torque: %.2f', torque_values(i)));

        % % Add a cross marker at the global maxima
        % hold on;
        % plot(sigma_values(max_sigma_idx), k_values(max_k_idx), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
        % hold off;
    end

    % Compute the average heatmap
    avg_heatmap = mean(heatmaps, 3);

    % Plot the averaged heatmap
    figure;
    imagesc(sigma_values, k_values, avg_heatmap);
    colorbar;
    xlabel('Sigma');
    ylabel('k');
    title('Averaged Cross-Correlation Heatmap');
    colormap('summer');
    grid on;

    % Final 2D plot of k vs sigma with torque as color gradient
    figure;
    scatter(maxima_sigma, maxima_k, 50, torque_values, 'filled'); % Color gradient based on torque
    xlabel('Sigma');
    ylabel('k');
    title('Maxima of k vs Sigma with Torque Gradient');
    colorbar;
    colormap('summer'); % Use a jet colormap for better visualization
    grid on;
end

function env = calculate_envelope(data, smooth_coeff)
    % Extend the signal to reduce edge effects
    data_ext = [data; data; data];  % Triplicate the data

    % Compute the envelope of the extended signal
    [env_ext, ~] = envelope(data_ext, 50, 'peak');

    % Extract only the middle section to avoid edge artifacts
    N = length(data);
    env = env_ext(N+1:2*N);

    % Smooth the envelope
    windowSize = 5;
    b = (1/windowSize) * ones(1, windowSize);
    a = 1;
    env = filter(b, a, env);

    % Normalize the envelope
    env = (env - min(env)) / (max(env) - min(env));
end

function [corr_score, env1, env2] = envelope_correlation(data1, data2, smooth_coeff, env_fig, ignore_shift)
    if nargin < 5
        ignore_shift = false; % Default to not ignoring shifts
    end

    % Compute envelopes using the new function
    env1 = calculate_envelope(data1, smooth_coeff);
    env2 = calculate_envelope(data2, smooth_coeff);


    % Compute correlation
    if ignore_shift
        % Use cross-correlation to find the best alignment
        [cross_corr, lags] = xcorr(env1, env2, 'coeff');
        [~, max_idx] = max(cross_corr);
        shift = lags(max_idx);
        env2 = circshift(env2, shift); % Align env2 to env1
    end

    % Refresh envelope/data plot
    figure(env_fig);
    clf(env_fig);
    hold on;
    plot(normalize(data1));
    plot(normalize(data2));
    plot(env1);
    plot(env2);
    pause(1);

    corr_score = corr(env1, env2, 'Type', 'Spearman');  
end