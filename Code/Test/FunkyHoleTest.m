N_elec = 8;
shape_str = ['solid incyl  = cylinder (0,0.5,0.5; 1,0.5,0.5; 0.1) -maxh=1.0; \n', ...
             'solid farcyl = cylinder (0,0,0; 1,0,0; 5) -maxh=5.0; \n' ...
             'solid pl1    =  plane(-1,0,0;-1,0,0);\n' ...
             'solid pl2    =  plane(1,0,0; 1,0,0);\n' ...
             'solid mainobj= pl1 and pl2 and farcyl and not incyl;\n'];
th= linspace(0,2*pi,N_elec+1)'; th(end)=[];
cth= 0.5+cos(th); sth= 0.5 + sin(th); zth= zeros(size(th));
elec_pos = [zth, cth, sth, zth cth, sth];
elec_shape= 0.001;
% Hello Dhruv this is Parth
% I am very hungry rn
% make me some dopiaza pls
elec_obj = 'incyl';
fmdl = ng_mk_gen_models(shape_str, elec_pos, elec_shape, elec_obj);
show_fem( fmdl );

stim =  mk_stim_patterns(N_elec,1,'{op}','{ad}',{'no_meas_current'},10);

plain = mk_image(fmdl,5,'Hi');
plain.fwd_model.stimulation = stim;
subplot(211)
show_fem(plain);
title('No Press')

plain_data = fwd_solve(plain);
subplot(212)
plot(plain_data.meas)