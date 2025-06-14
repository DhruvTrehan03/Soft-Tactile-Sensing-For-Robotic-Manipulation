%% Toggle Settings

initialise = 0;
load_data = 1;       % Set to true to reload data
find_shift =  1;      % Set to true to find the best shift
show_first_three_subplots = 0; % Set to false to hide first 3 subplots



%% Initialise EIDORS (Only really need to do this the first time you run eacch session)

if initialise
    clear
    load_data = 1;       % Set to true to reload data
    find_shift = 0;      % Set to true to find the best shift
    show_first_three_subplots = 1; % Set to false to hide first 3 subplots
    run('Source/eidors-v3.11-ng/eidors/eidors_startup.m'); % Initialize EIDOR

    height = 0;
    width = 4.4;
    len = 3.6;
    xy = [0 0 width width; 0 len len 0]';
    curve_type = 1;
    maxsz = 0.1;

    trunk_shape = {height, xy, curve_type, maxsz};
    elec_pos  = [32, 1.1];
    elec_shape = 0.2;

    mdl = ng_mk_extruded_model(trunk_shape, elec_pos, elec_shape);
    stim = mk_stim_patterns(32, 1, [0, 16], [0, 1], {'no_meas_current'}, 5);

    plain = mk_image(mdl, 1, 'Hi');
    plain.fwd_model.stimulation = stim;
end

%% Load Data (Only If Needed)
if load_data
    data_objs = load("SavedVariables\TorqueSlice.mat").clipped_data';
    data_objs = data_objs(data_objs ~= 0); % Remove zero values
    data_homg = load("SavedVariables\TorqueSliceHom.mat").clipped_data_hom';
    data_homg = data_homg(data_homg ~= 0); % Remove zero values
    data_diff = abs(data_objs - data_homg);
    data_diff = normalize(data_diff);
    windowSize = 5;
    b = (1/windowSize)*ones(1,windowSize);
    a = 1;
    % plot(clip(normalize(up,'center',0,'scale',1),-0.1,3))
    data_diff = filter(b,a,data_diff);
    % Apply an additional manual shift
    % data_diff = circshift(data_diff, 10*32);
end

%% Find Best Shift (If Needed)
if find_shift

    best_shift = find_best_shift_envelope(data_diff, sim_diff, 1, 896);
    data_diff = circshift(data_diff, best_shift);
end

%% Generate Models and Apply Function

functions = {'Step', ...                            1
            'Linear', ...                           2
            'Differential of a Gaussian (DoG)', ... 3
            'Modulated Gaussian (MG)', ...          4
            'Linear With Cut-Off'}; ...                5
function_choice = 5; 
%% Plot different models
plot_fem_and_cross_section(mdl,functions);

press = plain;
func = apply_function((function_choice));
press.elem_data = 1 + elem_select(press.fwd_model, func);
press.fwd_model.stimulation = stim;
% show_fem(press);

%% Plot Results
plain_data = fwd_solve(plain);
press_data = fwd_solve(press);
sim_diff = abs(press_data.meas - plain_data.meas);
sim_diff = normalize(sim_diff);
[correlation,env1,env2] = envelope_correlation(data_diff, sim_diff);



subplot_idx = 1;
if show_first_three_subplots
    h1 = subplot(5, 2, subplot_idx);
    show_fem(plain);
    title('No Press')
    subplot_idx = subplot_idx + 1;

    h2 = subplot(5, 2, subplot_idx);
    show_fem(press);
    title('Press')
    press.calc_colours.cb_shrink_move = [.3, .8, -0.02];
    common_colourbar([h1, h2], press);
    subplot_idx = subplot_idx + 1;

    subplot(5, 2, subplot_idx);
    plot(abs(plain_data.meas), 'b')
    title('No Press Electrodes')
    subplot_idx = subplot_idx + 1;

    subplot(5, 2, subplot_idx);
    plot(abs(press_data.meas), 'b')
    title('Press Electrodes')
    subplot_idx = subplot_idx + 1;

    subplot(5, 2, [subplot_idx, subplot_idx + 1]);
    plot(normalize(sim_diff,'center',0,'scale',1))
    title([sprintf('Electrodes Diff Sim (%s)',string(functions(function_choice)))])
    
    subplot(5, 2, [subplot_idx + 2, subplot_idx + 3]);
    hold on;
    plot(normalize(data_diff,'center',0,'scale',1))
    hold off;
    title('Electrodes Diff Real')
    
    subplot(5, 2, [subplot_idx + 4, subplot_idx + 5])
    text(0.45, 0.5, num2str(correlation), "FontSize", 16);
    axis off;
    title({'Correlation Score'})
else
    % Define smoothing window size
    smooth_window = 30;  % Adjust as needed
    
    subplot(3, 2, [subplot_idx, subplot_idx + 1]);
    % plot(data_diff, 'b', 'DisplayName', 'Data Difference'); hold on;  % Original data in blue
    plot(env1, 'r', 'LineWidth', 1.5,'DisplayName','Data Difference Envelope'); % Smoothed envelope in red
    hold off;
    title('Data Difference with Envelope');
    legend();
    
    subplot(3, 2, [subplot_idx + 2, subplot_idx + 3]);
    % plot(sim_diff, 'b','DisplayName', 'Simulation Difference'); hold on;  % Simulation difference in blue
    plot(env2, 'r', 'LineWidth', 1.5, 'DisplayName','Simulation Difference Envelope'); % Smoothed envelope in red
    hold off;
    title('Simulation Difference with Envelope');
    legend();
    
    subplot(3, 2, [subplot_idx + 4, subplot_idx + 5])
    text(0.45, 0.5, num2str(correlation), "FontSize", 16);
    axis off;
    title({'Correlation Score'})
end


%% Function: Finding Best Shift
function best_shift = find_best_shift(data_diff, sim_diff, shift_step, max_shifts)
    biggest_cor = [0, 0]; % [correlation, shift index]
    corrs = zeros([max_shifts, 1]);

    for i = 1:max_shifts
        shifted_data = circshift(data_diff, i * shift_step);
        correlation = corr(shifted_data, sim_diff);
        corrs(i) = correlation;
        if correlation > biggest_cor(1)
            biggest_cor = [correlation, i];
        end
    end

    best_shift = biggest_cor(2) * shift_step;
    fprintf('Best shift: %d samples, Correlation: %.4f\n', best_shift, biggest_cor(1));
end

%% Function: Applying Selected Function to Model
function select_fcn = apply_function(choice)
    centre = 1.8;
    

    switch lower(choice)
        case 1      %"step"
            select_fcn = @(x, y, z) 0.5 * (y > centre) - 0.5 * (y <= centre);
        case 2      %"linear"
            sigma = 3;
            select_fcn = @(x, y, z) (y - centre) / sigma;
        case 3      %"differential of a gaussian"
            sigma = 0.4;
            select_fcn = @(x, y, z) -(centre-y) .* exp(-((centre-y).^2) / (2 * sigma^2)) / (sigma^2);
        case 4      %"modulated gaussian"
            sigma = 0.4;
            k = 5;
            select_fcn = @(x,y,z) exp(-(y - centre).^2 / (2 * sigma^2)) .*(cos(k * (y-centre)));
        case 5      %modulated difference of a gaussian
            sigma = 4;
            k = 3;
            % select_fcn = @(x,y,z) ((abs(y - centre) < k/2).* exp(-1 ./ (1 - ((2*(y - centre)/k).^2))) .* (1 + sigma * sin(pi*(y - centre)/k)) );
select_fcn = @(x, y, z) ( (abs(y-centre) < k/2) ...
    .* (0.5 * (1 + cos((2*pi/k) * (y-centre)))) ...
    .* (1 + sigma * sin(pi*(y-centre)/k)) );



        otherwise
            error("Unknown function choice: %s", choice);
    end
    % figure()
    % plot(select_fcn(0,linspace(0,3.6,100),0))
    % figure()
end

function plot_fem_and_cross_section(mdl, function_choices)
    % Plots FEM models alongside their cross-sectional conductivity profiles
    %
    % Inputs:
    %   mdl              - FEM model
    %   params           - Structure containing function parameters
    %   function_choices - Cell array of function choices to compare
    %
    num_models = length(function_choices);
    figure;
    sgtitle('FEM Models and Cross-Sections');
    
    % X-values and Y-values for the cross-section
    x_vals = linspace(0, 4.4, 100);
    y_vals = linspace(0, 3.6, 100); % Define y-values range
    
    for i = 1:num_models
        
        function_choice = function_choices{i};
        select_fcn = apply_function(i); % Get the selection function
        img = mk_image(mdl, 5); % Create a homogeneous image
        img.elem_data = 5 + 10 * elem_select(img.fwd_model, select_fcn); % Apply inhomogeneity

        % **Top row: FEM Model**
        subplot(2, num_models, i);
        % figure()
        show_fem(img);
        title(sprintf('FEM: %s', strrep(function_choice, '_', ' ')));
        axis tight;

        % **Bottom row: Cross-section plot**
        subplot(2, num_models, num_models + i);
        % figure();
        conductivity_values = select_fcn(x_vals,y_vals,0);
        plot(y_vals, conductivity_values, 'LineWidth', 2);
        xlabel('Y Position');
        ylabel('Conductivity (S/m)');
        title(sprintf('Cross-Section: %s', strrep(function_choice, '_', ' ')));
        grid on;
        xlim([0,3.6]);
    end
    figure();
end


function best_shift = find_best_shift_envelope(data_diff, sim_diff, shift_step, max_shifts)
    % Function to find the best shift using envelope correlation
    
    max_corr = -Inf;
    best_shift = 0;

    for i = 1:max_shifts
        shifted_data = circshift(data_diff, i * shift_step);
        score = envelope_correlation(shifted_data, sim_diff);  % Use the envelope correlation function

        if score > max_corr
            max_corr = score;
            best_shift = i * shift_step;
        end
    end

    fprintf('Best shift: %d samples (Envelope Correlation: %.4f)\n', best_shift, max_corr);
end

function [corr_score,env1,env2] = envelope_correlation(data1, data2)
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
