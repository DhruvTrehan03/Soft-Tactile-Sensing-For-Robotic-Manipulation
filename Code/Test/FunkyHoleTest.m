% clear
% run('Source/eidors-v3.11-ng/eidors/eidors_startup.m'); % Initialize EIDORS
% 
% N_elec = 8;
% offset = input('Offset: ');
% shape_str = sprintf(['solid incyl  = cylinder (0,%.2f,0; 1,%.2f,0; 1) -maxh=1.0; \n', ...
%                      'solid farcyl = cylinder (0,0,0; 1,0,0; 5) -maxh=5.0; \n' ...
%                      'solid pl1    =  plane(-1,0,0;-1,0,0);\n' ...
%                      'solid pl2    =  plane(1,0,0; 1,0,0);\n' ...
%                      'solid mainobj= pl1 and pl2 and farcyl and not incyl;\n'], offset, offset);
% shape_str = sprintf(['solid mainbox = orthobrick(-5, -5, -0.5; 5, 5, 0.5);\n', ...
%                      'solid cut1    = orthobrick(-2, -1, -1; 2, -0.5, 1);\n', ...
%                      'solid cut2    = orthobrick(-2, 0.5, -1; 2, 1, 1);\n', ...
%                      'solid mainobj = mainbox and not cut1 and not cut2;\n']);
% 
% 
% 
% th= linspace(0,2*pi,N_elec+1)'; th(end)=[];
% cth= offset+cos(th); sth= sin(th); zth= zeros(size(th));
% elec_pos = [zth, cth, sth, zth, cth, sth];
% elec_shape= 0.001;
% elec_obj = 'incyl';
% fmdl = ng_mk_gen_models(shape_str, elec_pos, elec_shape, elec_obj);
% show_fem( fmdl );
% 
% stim =  mk_stim_patterns(N_elec,1,'{op}','{ad}',{'no_meas_current'},10);
% 
% plain = mk_image(fmdl,5,'Hi');
% plain.fwd_model.stimulation = stim;
% subplot(211)
% show_fem(plain);
% title('No Press')
% 
% plain_data = fwd_solve(plain);
% subplot(212)
% plot(plain_data.meas)
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

stim = mk_stim_patterns(2*n_elec, 1, [0, 1], [1, 0], {'no_meas_current'}, 5);
r=0.3;
select_fcn = @(x,y,z)((x-2.2).^2+(y-1.5).^2<r^2);
% select_fcn = @(x,y,z)(x>1.9) & (x<2.5) & (y<1.8) & (y>1.2);

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