%% Toggle Settings

initialise = 0;
load_data = 0;       % Set to true to reload data
find_shift = 0;      % Set to true to find the best shift
generate_model = 0;  % Set to true to generate the EIDORS model
show_first_three_subplots = 0; % Set to false to hide first 3 subplots
function_choice = "linear"; % Choose function: "differential of a gaussian", "step", "linear", "modulated gaussian"



%% Initialise EIDORS (Only really need to do this the first time you run eacch session)

if initialise
    clear
    load_data = 1;       % Set to true to reload data
    find_shift = 0;      % Set to true to find the best shift
    generate_model = 1;  % Set to true to generate the EIDORS model
    show_first_three_subplots = 1; % Set to false to hide first 3 subplots
    run('Source/eidors-v3.11-ng/eidors/eidors_startup.m'); % Initialize EIDOR
end

%% Load Data (Only If Needed)
if load_data
    data_objs = load("SavedVariables\TorqueSlice.mat").clipped_data';
    data_objs = data_objs(data_objs ~= 0); % Remove zero values
    data_homg = load("SavedVariables\TorqueSliceHom.mat").clipped_data_hom';
    data_homg = data_homg(data_homg ~= 0); % Remove zero values
    data_diff = abs(data_objs - data_homg);
    % Apply an additional manual shift
    data_diff = circshift(data_diff, 10*32);
end

%% Find Best Shift (If Needed)
if find_shift
    best_shift = find_best_shift(data_diff, sim_diff, 32, 28);
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
press.elem_data = 1 + apply_function(function_choice, press.fwd_model);
press.fwd_model.stimulation = stim;
show_fem(press);
Model_Compare = figure("Name", 'Model Comparison');
%% Plot Results
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
    plain_data = fwd_solve(plain);
    press_data = fwd_solve(press);
    plot(abs(plain_data.meas), 'b')
    title('No Press Electrodes')
    subplot_idx = subplot_idx + 1;

    subplot(5, 2, subplot_idx);
    plot(abs(press_data.meas), 'b')
    title('Press Electrodes')
    subplot_idx = subplot_idx + 1;

    subplot(5, 2, [subplot_idx, subplot_idx + 1]);
    sim_diff = abs(press_data.meas - plain_data.meas);
    plot(sim_diff)
    title([sprintf('Electrodes Diff Sim (%s)',function_choice)])
    
    subplot(5, 2, [subplot_idx + 2, subplot_idx + 3]);
    plot(data_diff)
    title('Electrodes Diff Real')
    
    subplot(5, 2, [subplot_idx + 4, subplot_idx + 5])
    correlation = corr(data_diff, sim_diff);
    text(0.45, 0.5, num2str(correlation), "FontSize", 16);
    axis off;
    title({'Correlation Score'})
else
    subplot(3, 2, [subplot_idx, subplot_idx + 1]);
    sim_diff = abs(press_data.meas - plain_data.meas);
    plot(sim_diff)
    title([sprintf('Electrodes Diff Sim (%s)',function_choice)])
    
    subplot(3, 2, [subplot_idx + 2, subplot_idx + 3]);
    plot(data_diff)
    title('Electrodes Diff Real')
    
    subplot(3, 2, [subplot_idx + 4, subplot_idx + 5])
    correlation = corr(data_diff, sim_diff);
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
function elem_data = apply_function(choice, fwd_model)
    centre = 1.8;
    sigma = 1;

    switch lower(choice)
        case "differential of a gaussian"
            select_fcn = @(x, y, z) -(y - centre) .* exp(-((y - centre).^2) / (2 * sigma^2)) / (sigma^2);
        case "step"
            select_fcn = @(x, y, z) 0.5 * (y > centre) - 0.5 * (y <= centre);
        case "linear"
            select_fcn = @(x, y, z) (y - centre) / sigma;
        case "modulated gaussian"
            sigma = 0.2;
            k = 10;
            scale = 0.1;
            select_fcn = @(x,y,z) exp(-(y - centre).^2 / (2 * sigma^2)) .*(scale * cos(k * (y-centre)));
        otherwise
            error("Unknown function choice: %s", choice);
    end
    figure()
    plot(select_fcn(0,linspace(0,3.6,100),0))
    figure()
    elem_data = elem_select(fwd_model, select_fcn);
end
