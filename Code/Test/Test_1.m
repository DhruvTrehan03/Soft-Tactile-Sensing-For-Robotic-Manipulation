% centre = 1.8;
% sigma = 0.2;
% scale = 0.1;
% k=10;
% 
% 
% select_fcn = @(x,y,z) exp(-(y - centre).^2 / (2 * sigma^2)) .*(scale * cos(k * (y-centre)));
% 
% plot(select_fcn(0,linspace(0,3.6,100),0))

clear
run('Source/eidors-v3.11-ng/eidors/eidors_startup.m'); % Initialize EIDORS

%% Define the main rectangle (outer boundary)
width  = 4.4;
height = 3.6;
outer_xy = [0,  0;
            width,  0;
            width, height;
            0,  height];  % Counterclockwise order

%% Define two inner rectangular holes (must be counterclockwise)
hole1_xy = [1,   1;
            1.5, 1;
            1.5, 2;
            1,   2];  % First hole (mini-rectangle)

hole2_xy = [2.9, 1;
            3.4, 1;
            3.4, 2;
            2.9, 2]; % Second hole (mini-rectangle)

%% Define the complete shape with holes
shape = {outer_xy, hole1_xy, hole2_xy, 0.1};  % Last element is max mesh size

%% Define electrodes
elec_outer   = [0, 1];  % Electrodes on outer boundary (even spacing)
n_elec=12;
elec_inner1  = [1.5 * ones(n_elec,1), linspace(1,2,n_elec)'];  % 4 electrodes on hole1
elec_inner2  = [2.9 * ones(n_elec,1), linspace(1,2,n_elec)'];  % 4 electrodes on second mini-rectangle

elec_pos = {elec_outer, elec_inner1, elec_inner2}; 

%% Generate the model
mdl = ng_mk_2d_model(shape, elec_pos);

stim = mk_stim_patterns(2*n_elec, 1, [1, 0], [0, 1], {'no_meas_current'}, 5);
r=0.3;
select_fcn = @(x,y,z)((x-2.2).^2+(y-1.5).^2<r^2);
select_fcn = @(x,y,z)(x>1.9) & (x<2.5) & (y<1.8) & (y>1.2);

%% Generate Models and Apply Function
plain = mk_image(mdl, 10, 'Hi');
plain.fwd_model.stimulation = stim;
plain.fwd_solve.get_all_meas = 1 ;
plain_data  = fwd_solve(plain);
plain_volts = rmfield(plain, 'elem_data');
plain_volts.node_data = plain_data.volt(:,1);

ball = mk_image(mdl,1,'Bye');
ball.fwd_model.stimulation = stim;
ball.elem_data = 10+0.1*elem_select(mdl, select_fcn);
ball.fwd_solve.get_all_meas = 1 ;
ball_data = fwd_solve(ball);
ball_volts = rmfield(ball, 'elem_data');
ball_volts.node_data = ball_data.volt(:,1);

%% Create a single figure with tight layout
figure();
tiledlayout(2,3, 'TileSpacing', 'Compact', 'Padding', 'Compact');

% Subplot 1: FEM model without perturbation
nexttile;
show_fem(plain);
title('Base FEM Model');

% Subplot 2: FEM model with perturbation
nexttile;
show_fem(ball);
title('FEM Model with Perturbation');

% Subplot 3: Difference in voltage distribution
nexttile;
ball_volts.node_data = ball_volts.node_data - plain_volts.node_data;
show_fem_enhanced(ball_volts);
title('Voltage Difference');

% Subplot 4: Measurement data for plain model
nexttile;
plot(plain_data.meas);
title('Measurement Data (Plain Model)');
xlabel('Measurement Index');
ylabel('Voltage');

% Subplot 5: Measurement data for ball model
nexttile;
plot(ball_data.meas);
title('Measurement Data (Perturbed Model)');
xlabel('Measurement Index');
ylabel('Voltage');

% Subplot 6: Absolute difference in measurement data
nexttile;
plot(abs(ball_data.meas - plain_data.meas));
title('Absolute Difference in Measurements');
xlabel('Measurement Index');
ylabel('Voltage Difference');

%% Final adjustments
sgtitle('Electrical Impedance Tomography (EIT) Analysis');
