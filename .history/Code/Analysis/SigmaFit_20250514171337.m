clear; close all; clc;

if isempty(which('eidors_startup'))
    run('Source/eidors-v3.11-ng/eidors/eidors_startup.m'); % Initialize EIDORS
end

load("Analysis\ModelGen.mat", "mdl", "stim", "plain");
    plain_data = fwd_solve(plain);
    % Define parameters
    base_dir = 'SavedVariables\Diameter_Slices';
    diameters = {'10mm', '20mm', '30mm', '40mm'};
    centre = 1.8;
    k_const = 4;
    sigma_values = linspace(0.01, 4.4, 10);
    smooth_coeff = 50;

    % Define different select_fcn options
    select_fcns = {
        %@( y, s) exp(-(y - centre).^2 / (2 * s^2)) .* cos(k_const * (y - centre)),  % Modulated Gaussian
        %@( y, s) -(y - centre) .* exp(-((y - centre).^2) / (2 * s^2)) / (s^2),     % Difference of Gaussians
        %@(y,s)(y>centre-k_const/2).*(y<centre+k_const/2).*(y-centre)/s, %linear with k
        @(y, s) (y <= centre) .* exp(-(y - centre).^2 / (2 * s^2)) + (y > centre) .* exp(-(y - centre).^2 / (2 * s^2)) .* cos(k_const * (y - centre))% Modulated Gaussian with step

    };

    % Loop through select_fcn options
    for i = 1:length(select_fcns)
        fprintf('Processing with select_fcn #%d...\n', i);
        SFit(base_dir, diameters, mdl, stim,plain,plain_data, select_fcns{i}, sigma_values, smooth_coeff, 'sigma_corr_matrix.mat');
    end



function SFit(base_dir, diameters, mdl, stim,plain, plain_data, select_fcn, sigma_values, smooth_coeff, save_path)
    % SIGMAFIT: Computes and saves the correlation matrix for sigma values.

    % Check if a saved correlation matrix exists
    if isfile(save_path)
        load(save_path, 'correlation_matrix', 'existing_sigma_values');
    else
        correlation_matrix = [];
        existing_sigma_values = [];
    end

    % Find new sigma values to calculate
    [new_sigma_values, new_indices] = setdiff(sigma_values, existing_sigma_values);

    % Initialize storage for new correlations
    new_correlations = zeros(length(diameters), length(new_sigma_values));

    % Initialize figures outside the loops
    env_fig = figure('Name', 'Envelope Plot', 'NumberTitle', 'off','Position', [600, 100, 480, 360]);
    fem_fig = figure('Name', 'FEM Plot', 'NumberTitle', 'off','Position', [100, 100, 480, 360]);

    % Load the precomputed matrix for varying k and sigma
    load('sim_data_matrix.mat', 'sim_data_matrix','k_values','sigma_values');
    % Initialize heatmap storage for averaging
    average_heatmap_data = zeros(length(k_values), length(sigma_values), length(diameters));

    % Loop through each diameter
    for d = 1:length(diameters)
        diameter = diameters{d};
        data_dir = fullfile(base_dir);

        % Get list of data files
        files = dir(fullfile(data_dir, sprintf('%dmm_Data_*.mat', str2double(diameter(1:end-2)))));

        % Initialize heatmap accumulator and file counter
        heatmap_accumulator = zeros(length(k_values), length(sigma_values));
        file_count = 0;

        for i = 1:length(files)
            % Load and transpose data
            file_path = fullfile(files(i).folder, files(i).name);
            data_diff = load(file_path).data_diff';
            data_diff = smoothdata(data_diff, "gaussian", 9); % Smooth data_diff

            % Initialize progress bar
            progress_bar = waitbar(0, sprintf('Processing Diameter: %s, File: %d/%d', diameter, i, length(files)),"Position", [250, 450, 300, 60]);

            % Compute cross-correlation for each k and sigma
            for k_idx = 1:length(k_values)
                for sigma_idx = 1:length(sigma_values)
                    % Extract precomputed data for current k and sigma
                    sim_data = squeeze(sim_data_matrix(k_idx, sigma_idx, :));

                    % Compute cross-correlation
                    [cross_corr, ~] = xcorr(data_diff, sim_data, 'coeff');
                    heatmap_accumulator(k_idx, sigma_idx) = heatmap_accumulator(k_idx, sigma_idx) + max(cross_corr); % Accumulate maximum correlation
                end
            end

            % Close progress bar
            close(progress_bar);

            % Increment file counter
            file_count = file_count + 1;
        end

        % Compute the average heatmap for the current diameter
        average_heatmap_data(:, :, d) = heatmap_accumulator / file_count;

        % Generate heatmap for the current diameter
        figure;
        imagesc(sigma_values, k_values, average_heatmap_data(:, :, d));
        colorbar;
        xlabel('Sigma');
        ylabel('k');
        title(sprintf('Average Cross-Correlation Heatmap for Diameter: %s', diameter));

        % Find the global maxima
        [max_val, max_idx] = max(average_heatmap_data(:, :, d), [], 'all', 'linear');
        [max_k_idx, max_sigma_idx] = ind2sub(size(average_heatmap_data(:, :, d)), max_idx);

        % Add a cross marker at the global maxima
        hold on;
        plot(sigma_values(max_sigma_idx), k_values(max_k_idx), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
        hold off;
    end

    % Update the correlation matrix
    all_sigma_values = unique([existing_sigma_values, sigma_values]);
    updated_correlation_matrix = zeros(length(diameters), length(all_sigma_values));
    [~, existing_indices] = ismember(existing_sigma_values, all_sigma_values);
    [~, new_indices] = ismember(new_sigma_values, all_sigma_values);

    updated_correlation_matrix(:, existing_indices) = correlation_matrix;
    updated_correlation_matrix(:, new_indices) = new_correlations;

    % Save the updated correlation matrix
    correlation_matrix = updated_correlation_matrix;
    existing_sigma_values = all_sigma_values;

    % Plot the updated correlation matrix
    figure;
    hold on;
    for d = 1:length(diameters)
        plot(all_sigma_values, correlation_matrix(d, :), 'DisplayName', sprintf('%s Diameter', diameters{d}));
    end
    hold off;
    xlabel('Sigma');
    ylabel('Correlation');
    title('Updated Correlation vs Sigma');
    legend('Location', 'best');
    grid on;

    disp("Do you want to save the correlation matrix? (y/n)");
    answer = input('', 's');
    if strcmp(answer, 'y')
        % Save the updated correlation matrix
        save(save_path, 'correlation_matrix', 'existing_sigma_values');
    else
        disp('Not saving the correlation matrix.');
    end
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