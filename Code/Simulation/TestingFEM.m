%% Toggle Settings

initialise = 0;
load_data = 1;       % Set to true to reload data
find_shift =  1;      % Set to true to find the best shift
generate_model = 0;  % Set to true to generate the EIDORS model
show_first_three_subplots = 0; % Set to false to hide first 3 subplots
functions = {'Step', ...                            1
            'Linear', ...                           2
            'Differential of a Gaussian (DoG)', ... 3
            'Modulated Gaussian (MG)', ...          4
            'Experimental (whatever i feel like)'}; ...                5
function_choice = 4; 


%% Initialise EIDORS (Only really need to do this the first time you run eacch session)

if initialise
    clear
    load_data = 1;       % Set to true to reload data
    find_shift = 0;      % Set to true to find the best shift
    generate_model = 1;  % Set to true to generate the EIDORS model
    show_first_three_subplots = 1; % Set to false to hide first 3 subplots
    functions = {'Step', ...                            1
            'Linear', ...                           2
            'Differential of a Gaussian (DoG)', ... 3
            'Modulated Gaussian (MG)', ...          4
            'Antisymmetric MG'}; ...                5
    function_choice = 1; 
    run('Source/eidors-v3.11-ng/eidors/eidors_startup.m'); % Initialize EIDOR
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

    best_shift = find_best_shift(data_diff, sim_diff, 1, 896);
    data_diff = circshift(data_diff, best_shift);
end

%% Model Generation (Only If Needed)
if generate_model
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
end

%% Generate Models and Apply Function


plain = mk_image(mdl, 1, 'Hi');
plain.fwd_model.stimulation = stim;

press = plain;
press.elem_data = 1 + elem_select(press.fwd_model, apply_function((function_choice)));
press.fwd_model.stimulation = stim;
show_fem(press);
Model_Compare = figure("Name", 'Model Comparison');
%% Plot Results
plain_data = fwd_solve(plain);
press_data = fwd_solve(press);
sim_diff = abs(press_data.meas - plain_data.meas);
sim_diff = normalize(sim_diff);
correlation = corr(data_diff, sim_diff);
[up,lo] = envelope(sim_diff,10,'peak');

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
    plot(clip(normalize(up,'center',0,'scale',1),-0.1,5))
    plot(normalize(data_diff,'center',0,'scale',1))
    hold off;
    title('Electrodes Diff Real')
    
    subplot(5, 2, [subplot_idx + 4, subplot_idx + 5])
    text(0.45, 0.5, num2str(correlation), "FontSize", 16);
    axis off;
    title({'Correlation Score'})
else
    subplot(3, 2, [subplot_idx, subplot_idx + 1]);
    sim_diff = abs(press_data.meas - plain_data.meas);
    plot(sim_diff)
    title([sprintf('Electrodes Diff Sim (%s)',string(functions(function_choice)))])
    
    subplot(3, 2, [subplot_idx + 2, subplot_idx + 3]);
    hold on;
    % plot(clip(normalize(up,'center',0,'scale',1),-0.1,3))
    plot(data_diff)
    hold off;
    title('Electrodes Diff Real')
    
    subplot(3, 2, [subplot_idx + 4, subplot_idx + 5])
    text(0.45, 0.5, num2str(correlation), "FontSize", 16);
    axis off;
    title({'Correlation Score'})
end
%% Plot different models
plot_fem_and_cross_section(mdl,functions);

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
    sigma = 1;

    switch lower(choice)
        case 1      %"differential of a gaussian"
            select_fcn = @(x, y, z) -(y - centre) .* exp(-((y - centre).^2) / (2 * sigma^2)) / (sigma^2);
        case 2      %"step"
            select_fcn = @(x, y, z) 0.5 * (y > centre) - 0.5 * (y <= centre);
        case 3      %"linear"
            select_fcn = @(x, y, z) (y - centre) / sigma;
        case 4      %"modulated gaussian"
            sigma = 0.4;
            k = 5;
            select_fcn = @(x,y,z) exp(-(y - centre).^2 / (2 * sigma^2)) .*(cos(k * (y-centre)));
        case 5      %modulated difference of a gaussian
            sigma = 0.2;
            k = 4;
            select_fcn = @(x,y,z) (y>centre).*exp(-(y - centre).^2 / (2 * sigma^2)) .*(cos(k * (y-centre))) + 0.1* exp(-(y - centre).^2 / (2 * sigma^2));
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
        show_fem(img);
        title(sprintf('FEM: %s', strrep(function_choice, '_', ' ')));
        axis tight;

        % **Bottom row: Cross-section plot**
        subplot(2, num_models, num_models + i);
        conductivity_values = select_fcn(x_vals,y_vals,0);
        plot(y_vals, conductivity_values, 'LineWidth', 2);
        xlabel('Y Position');
        ylabel('Conductivity (S/m)');
        title(sprintf('Cross-Section: %s', strrep(function_choice, '_', ' ')));
        grid on;
        xlim([0,3.6]);
    end
end

