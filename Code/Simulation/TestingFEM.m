% Main Script
clear
run('Source/eidors-v3.11-ng/eidors/eidors_startup.m'); % Initialize EIDORS
figure();
height = 0;
width = 4.4;
len = 3.6;
[xy] = [0 0 width width; 0 len len 0]';
curve_type = 1;
maxsz= 0.1;

trunk_shape = { height,xy,curve_type,maxsz};
elec_pos  = [32,1.1];
elec_shape = 0.2; 

mdl = ng_mk_extruded_model(trunk_shape, elec_pos, elec_shape);
stim =  mk_stim_patterns(32,1,[0,16],[0,1],{'no_meas_current'}, 5);

plain = mk_image(mdl,1,'Hi');
plain.fwd_model.stimulation = stim;
h1 = subplot(3,2,1);
show_fem(plain);
title('No Press')
press = plain;

centre = 1.8;
sigma = 0.6;
% select_fcn = inline('(x-2.2).^2+(y-1.8).^2<0.4^2','x','y','z');
select_fcn = @(x, y, z) -(y - centre) .* exp(-((y - centre).^2) / (2 * sigma^2)) / (sigma^2);
press.elem_data = 1 + elem_select(press.fwd_model, select_fcn);
press.fwd_model.stimulation = stim;

h2 = subplot(3,2,2);
show_fem(press);
title('Press')
press.calc_colours.cb_shrink_move = [.3,.8,-0.02];
common_colourbar([h1,h2],press);
subplot(3,2,3);
plain_data = fwd_solve(plain);
press_data = fwd_solve(press);
plot(abs(plain_data.meas),'b')
title('No Press Electrodes')
subplot(3,2,4);
plot(abs(press_data.meas),'b')
title('Press Electrodes')
subplot(3,2,[5,6]);
plot(abs(press_data.meas-plain_data.meas))
title('Electrodes Diff')
