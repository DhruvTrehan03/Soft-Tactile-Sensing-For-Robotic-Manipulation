% Main Script
% figure()
% subplot(3,1,1)
% plot(data_hom)
% subplot(3,1,2)
% plot(data_objs)
% subplot(3,1,3)
% plot(abs(data_objs-data_hom))

clear
[data_hom, data_objs]= data_slice();
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

imdl = hyperparameter(imdl,3);

imdl.name= 'dsh_rect'; 
imdl= eidors_obj('inv_model', imdl);
valid_inv_model(imdl);
[stim, meas_select] = mk_stim_patterns(32,1,[0,16],[0,1],{'no_meas_current'}, 5);
imdl.fwd_model.stimulation = stim;
imdl.fwd_model.meas_select = meas_select;
img = inv_solve(imdl, data_hom, data_objs);
show_slices(img);


function [data_homg, data_objs] = data_slice()
% Prompt the user to select a directory
    [selected_file,location] = uigetfile('C:\\Users\\dhruv\\Soft-Tactile-Sensing-For-Robotic-Manipulation\\Readings\\', 'Select a directory containing .mat files');
    if selected_file ~= 0  % Check if the user didn't cancel the dialog
       file_name = fullfile(location,selected_file);
       disp(file_name)
       data = load(file_name);
       % Get all variable names stored in the .mat file
        varNames = fieldnames(data);

        % If there is only one variable in the file, extract it
        if isscalar(varNames)
            data = data.(varNames{1});
        else
            error('Multiple variables found in the file. Specify which one to use.');
        end

    else
        disp('No directory was selected.');
    end
    h1 = figure();
    plot(data(:,2:end))
    x = input('Index for homg data: ');
    data_homg = data(x,2:end)';
    y = input('Index for objs data: ');
    data_objs = data(y,2:end)';
end

function[inv2d] =hyperparameter(inv2d,n)
    if n==1
    % case 1
        % inv2d.hyperparameter.value = 0.004; % Lorcan
        inv2d.hyperparameter.value = 0.04;
        inv2d.solve=       'inv_solve_diff_GN_one_step';
        inv2d.RtR_prior=   'prior_laplace';
    elseif n==2
        % case 2
        inv2d.hyperparameter.value = 1e-2; % Lorcan
        % inv2d.hyperparameter.value = 1e-1;
        inv2d.RtR_prior=   'prior_laplace';
        inv2d.solve=       'np_inv_solve';
    elseif n==3
        % case 3
        inv2d.hyperparameter.func = @choose_noise_figure;
        inv2d.hyperparameter.noise_figure=2;
        inv2d.hyperparameter.tgt_elems= 1:6;
        inv2d.RtR_prior=   'prior_gaussian_HPF';
        inv2d.solve=       'inv_solve_diff_GN_one_step';
    elseif n==31
        %case 3.1
        inv2d.hyperparameter.func = @choose_noise_figure;
        inv2d.hyperparameter.noise_figure= 2;
        inv2d.hyperparameter.tgt_elems= 1:4;
        inv2d.RtR_prior=   @prior_laplace;
        inv2d.solve=       'inv_solve_diff_GN_one_step';
    elseif n==4
        % case 4
        inv2d.hyperparameter.value = 1e-6;
        inv2d.parameters.max_iterations= 10;
        inv2d.R_prior=     'prior_TV';
        inv2d.solve=       'inv_solve_TV_pdipm';
        inv2d.parameters.keep_iterations=1;
    elseif n==5
        % case 5
        inv2d.hyperparameter.value = 1e-4;
        inv2d.solve=       'aa_inv_total_var';
        inv2d.R_prior=     'prior_laplace';
        inv2d.parameters.max_iterations= 10;
    elseif n==6
        % case 6
        ;
    elseif n==7
        % case 7
        inv2d.hyperparameter.value = 1e-2;
        inv2d.parameters.max_iterations = 1e3;
        inv2d.parameters.term_tolerance = 1e-3;
        inv2d.solve=          'aa_inv_conj_grad';
        inv2d.R_prior=        'prior_TV';
    elseif n==8
        % case 8
        inv2d.hyperparameter.value = 1e-5;
        inv2d.parameters.max_iterations= 20;
        inv2d.R_prior=     'prior_TV';
        inv2d.solve=       'inv_solve_TV_irls';
        inv2d.parameters.keep_iterations=1;
    end


end
