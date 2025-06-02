clear; close all; clc;

% Define parameters
% base_dir = 'SavedVariables\Torque_Slices';
% TFit(base_dir);
base_dir = 'SavedVariables\Torque_Slices_Grouped';
TFit(base_dir);

function TFit(base_dir)
    % Load all data slice files
    files = dir(fullfile(base_dir, 'Data_Slice_*.mat'));
    torque_values = zeros(1, length(files));
    maxima_k = zeros(1, length(files));
    maxima_sigma = zeros(1, length(files));
    maxima_corr = zeros(1, length(files));
    heatmaps = cell(1, 10);
    % Load the precomputed matrix for varying k and sigma
    load('sim_data_matrix_focused.mat', 'sim_data_matrix', 'k_values', 'sigma_values');
    waitbar_handle = waitbar(0, 'Processing Data Slices...');
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
        % if torque value has been seen before add it to coresponding heatmaps slot
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



        % Find the global maxima
        [max_val, max_idx] = max(heatmap_accumulator, [], 'all', 'linear');
        [max_k_idx, max_sigma_idx] = ind2sub(size(heatmap_accumulator), max_idx);

        % Store the maxima for final plot
        maxima_k(i) = k_values(max_k_idx);
        maxima_sigma(i) = sigma_values(max_sigma_idx);
        maxima_corr(i) = max_val;

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
    close(waitbar_handle);
    % Final 2D plot of k vs sigma with torque as color gradient
    figure;
    scatter3(maxima_sigma, maxima_k, torque_values, 50, torque_values, 'filled'); % Color gradient based on torque
    xlabel('Sigma');
    ylabel('k');
    zlabel('Torque Value');
    title('Maxima of k vs Sigma with Torque Gradient');
    colorbar;
    grid on;
    save('maxima_data.mat', 'maxima_k', 'maxima_sigma', 'torque_values', 'maxima_corr');
    disp("Average Correlation Score: " + max(maxima_corr));
end

