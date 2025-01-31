% Main Script
clear
run('Source/eidors-v3.11-ng/eidors/eidors_startup.m'); % Initialize EIDORS
data_objs = load("SavedVariables\TorqueSlice.mat").clipped_data';
data_hom = load("SavedVariables\TorqueSliceHom.mat").clipped_data_hom';
mdl = load("Simulation\Model.mat","mdl").mdl;

figure();
show_fem(mdl);

n_elec = [32, 1];
fmdl = load("Simulation\Model.mat","mdl").mdl;

[stm,els] =  mk_stim_patterns(32,1,[0,16],[0,1],{'no_meas_current'}, 5);


fmdl.stimulation = stm;
fmdl.meas_select = els;
fmdl.solve=      'eidors_default';
fmdl.system_mat= 'eidors_default';
fmdl.jacobian=   'eidors_default';
fmdl.normalize_measurements= 0;
mdl_2d   = eidors_obj('fwd_model', fmdl);
  
imdl= eidors_obj('inv_model', 'EIT inverse');
imdl.reconst_type= 'difference';
imdl.jacobian_bkgnd.value= 1;
imdl.fwd_model= mdl_2d;
%inv2d.fwd_model.np_fwd_solve.perm_sym= '{y}';
imdl.parameters.term_tolerance= 1e-4;

imdl.hyperparameter.value = 0.04;
imdl.solve=       'inv_solve_diff_GN_one_step';
imdl.RtR_prior=   'prior_laplace';

imdl.name= 'dsh_rect'; 
imdl= eidors_obj('inv_model', imdl);
valid_inv_model(imdl);
[stim, meas_select] = mk_stim_patterns(32,1,[0,16],[0,1],{'no_meas_current'}, 5);
imdl.fwd_model.stimulation = stim;
imdl.fwd_model.meas_select = meas_select;
img = inv_solve(imdl, data_hom, data_objs);
show_slices(img);

