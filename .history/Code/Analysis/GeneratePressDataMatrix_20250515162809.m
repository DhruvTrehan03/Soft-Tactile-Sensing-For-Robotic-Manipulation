clear; close all; clc;

if isempty(which('eidors_startup'))
    run('Source/eidors-v3.11-ng/eidors/eidors_startup.m'); % Initialize EIDORS
end

load("Analysis\ModelGen.mat", "mdl", "stim", "plain");
plain_data = fwd_solve(plain);
% Define parameters
base_dir = 'SavedVariables\Diameter_Slices';
k_values = linspace(2.9, 3.6, 10); % Define k values for the modulation
sigma_values = linspace(1.2, 2, 10);

    % Define different select_fcn options
select_fcns = @( y, s,k) -(y - k) .* exp(-((y - k).^2) / (2 * s^2)) / (s^2);     % Difference of Gaussians

sim_data_matrix = GenerateDataMatrix(mdl, stim, plain, plain_data, select_fcns, k_values, sigma_values);
% load('sim_data_matrix.mat', 'sim_data_matrix');
% disp(size(sim_data_matrix));
% Visualize the sim_data_matrix
VisualizeSimDataMatrix(sim_data_matrix, k_values, sigma_values);
run("Analysis\TorqueAnalysis.m");

function sim_data_matrix = GenerateDataMatrix(mdl, stim, plain,plain_data, select_fcn, k_values, sigma_values)
    % GENERATEPRESSDATAMATRIX: Generates a matrix of press_data over k and sigma.
    % Inputs:
    %   mdl         - Model structure
    %   stim        - Stimulation structure
    %   plain       - Plain model
    %   select_fcn  - Selection function
    %   k_values    - Array of k values
    %   sigma_values- Array of sigma values
    %   centre      - Centre value for the select function
    % Outputs:
    %   press_data_matrix - 3D matrix of press_data (k x sigma x measurements)

    % Initialize the matrix to store press_data
    num_k = length(k_values);
    num_sigma = length(sigma_values);
    num_meas = 896; % Number of measurements
    sim_data_matrix = zeros(num_k, num_sigma, num_meas);
    %Progress bar
    progress = waitbar(0, 'Generating Simulated Data Matrix...');
    % Loop through k and sigma values
    for i = 1:num_k
        k = k_values(i);
        for j = 1:num_sigma
            % Update progress bar
            waitbar((i-1)*num_sigma+j/(num_k*num_sigma), progress,'Generating Simulated Data Matrix...');
            sigma = sigma_values(j);

            % Update the plain model with the select function
            press = plain;
            press.elem_data = 1 + elem_select(press.fwd_model, ...
                @(x, y, z) select_fcn(y, sigma, k));
            press.fwd_model.stimulation = stim;

            % Solve the forward model
            press_data = fwd_solve(press);

            % Store the measurement data
            sim_data_matrix(i, j, :) = abs(press_data.meas-plain_data.meas); % Subtract the plain data to get the difference
        end
    end

    % Save the matrix to a file
    save('sim_data_matrix_focused.mat', 'sim_data_matrix', 'k_values', 'sigma_values');
    disp('Press data matrix saved to sim_data_matrix.mat');
end

function VisualizeSimDataMatrix(sim_data_matrix, k_values, sigma_values)
    % VISUALIZESIMDATAMATRIX: Displays the sim_data_matrix in a grid format.
    % Inputs:
    %   sim_data_matrix - 3D matrix of press_data (k x sigma x measurements)
    %   k_values        - Array of k values
    %   sigma_values    - Array of sigma values

    % Create a tiled layout for the plots
    figure;
    len = min(10, length(k_values)); % Set grid size to a maximum of 5x5 or less
    tiledlayout(len, len, 'TileSpacing', 'compact', 'Padding', 'compact');
    %Progress bar
    % progress = waitbar(0, 'Visualizing Simulated Data Matrix...');
    % Loop through the grid
    for i = 1:len
        for j = 1:len
            % Update progress bar
            % waitbar(((i-1)*len+j)/len^2, progress,'Visualizing Simulated Data Matrix...');
            % Get the corresponding k and sigma indices
            k_idx = round(linspace(1, length(k_values), len));
            sigma_idx = round(linspace(1, length(sigma_values), len));
            
            % Extract the data for the current cell
            sim_diff = squeeze(sim_data_matrix(k_idx(i), sigma_idx(j), :));
            
            % Create a subplot for the current cell
            nexttile;
            plot(sim_diff);
            % remove ticks
            set(gca, 'XTick', [], 'YTick', []);
            title(sprintf('k=%.2f, σ=%.2f', k_values(k_idx(i)), sigma_values(sigma_idx(j))));
            axis tight;
        end
    end
end
