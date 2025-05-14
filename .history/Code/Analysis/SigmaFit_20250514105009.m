    
    load("Analysis\ModelGen.mat", "mdl", "stim", "plain");
    plain_data = fwd_solve(plain);
    % Define parameters
    base_dir = 'SavedVariables\Diameter_Slices';
    diameters = {'10mm', '20mm', '30mm', '40mm'};
    centre = 1.8;
    k_const = 4;
    sigma_values = linspace(0.01, 2, 1000);
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
        SFit(base_dir, diameters, mdl, stim,plain, select_fcns{i}, sigma_values, smooth_coeff, 'sigma_corr_matrix.mat');
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

    % Loop through each diameter
    for d = 1:length(diameters)
        diameter = diameters{d};
        data_dir = fullfile(base_dir);

        % Get list of data files
        files = dir(fullfile(data_dir, sprintf('%dmm_Data_*.mat', str2double(diameter(1:end-2)))));

        for i = 1:length(files)
            % Load and transpose data
            file_path = fullfile(files(i).folder, files(i).name);
            data_diff = load(file_path).data_diff';
            data_diff = smoothdata(data_diff, "gaussian", 9); % Smooth data_diff

            % Initialize progress bar
            progress_bar = waitbar(0, sprintf('Processing Diameter: %s, File: %d/%d', diameter, i, length(files)));

            for j = 1:length(new_sigma_values)
                sigma = new_sigma_values(j);

                press = plain;
                press.elem_data = 1 + elem_select(press.fwd_model, @(x, y, z) exp(-(y - 1.8).^2 / (2 * sigma^2)));
                press.fwd_model.stimulation = stim;

                % Compute simulated difference
                press_data = fwd_solve(press);
                sim_diff = abs(press_data.meas - plain_data.meas);

                % Compute correlation
                new_correlations(d, j) = envelope_correlation(data_diff, sim_diff, smooth_coeff);

                % Update progress bar
                waitbar(j / length(new_sigma_values), progress_bar);
            end

            % Close progress bar
            close(progress_bar);
        end
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
    save(save_path, 'correlation_matrix', 'existing_sigma_values');

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
end
