% Must run EIDORS startup.m for reconstructions to work

figure();
data_objs = load("SavedVariables\TorqueSlice.mat").clipped_data';
data_homg = load("SavedVariables\TorqueSliceHom.mat").clipped_data_hom';
plotReconstruction(data_homg, data_objs);
sgtitle("A");

function plotReconstruction(data_homg, data_objs)
    % n_elec = 16;
    % xvec = linspace(-1,1,54);
    % yvec = linspace(-1,1,42);
    % fmdl = mk_grid_model([],xvec,yvec);
    
    n_elec = [32, 1];
    xy_size = [54, 42];
    xy_size = xy_size + 1;
    
    xvec = linspace(-1.25,1.25,xy_size(1));
    yvec = linspace(-1,1,xy_size(2));
    fmdl = mk_grid_model([],xvec,yvec);
    
    options = {'no_meas_current','no_rotate_meas'};
    
    % put 1/4 of elecs on each side
    tb_elecs= linspace(1, xy_size(1), 10); 
    %tb_elecs= tb_elecs(2:2:end);
    sd_elecs= linspace(1, xy_size(2), 8);
    %sd_elecs= sd_elecs(2:2:end);
    sd_elecs = sd_elecs(2:end-1);
    
    el_nodes= [];
    % Top nodes -left to right
    bdy_nodes= (1:xy_size(1)) + xy_size(1)*(xy_size(2)-1); 
    el_nodes= [el_nodes, bdy_nodes(tb_elecs)];
    % Right nodes - top to bottom
    bdy_nodes= (1:xy_size(2))*xy_size(1); 
    el_nodes= [el_nodes, bdy_nodes(fliplr(sd_elecs))];
    % Bottom nodes - right to left
    bdy_nodes= 1:xy_size(1); 
    el_nodes= [el_nodes, bdy_nodes(fliplr(tb_elecs))];
    % Left nodes - bottom to top
    bdy_nodes= (0:xy_size(2)-1)*xy_size(1)+1; 
    el_nodes= [el_nodes, bdy_nodes(sd_elecs)];
    
    for i=1:n_elec(1)
        n = el_nodes(i);
        fmdl.electrode(i).nodes= n;
        fmdl.electrode(i).z_contact= 0.001; % choose a low value
    end
    
    
    subplot(1,5,1)
    show_fem(fmdl);
    
    subplot(1,5,2);
    imdl= add_params_2d_mdl(fmdl, n_elec(1), options, 1, data_homg, data_objs);
    
    subplot(1,5,3);
    imdl= add_params_2d_mdl(fmdl, n_elec(1), options, 2, data_homg, data_objs);
    
    % subplot(1,5,4);
    % imdl= add_params_2d_mdl(fmdl, n_elec(1), options, 3, data_homg, data_objs);
    % 
    % subplot(1,5,5);
    % imdl= add_params_2d_mdl(fmdl, n_elec(1), options, 31, data_homg, data_objs);
end



function inv2d= add_params_2d_mdl(params, n_elec, options, n, homg, objs)
    n_rings= 1;
    [st, els]= mk_stim_patterns(n_elec, n_rings, '{op}','{ad}', options, 10);
    params.stimulation= st;
    params.meas_select= els;
    params.solve=      'eidors_default';
    params.system_mat= 'eidors_default';
    params.jacobian=   'eidors_default';
    params.normalize_measurements= 0;
    mdl_2d   = eidors_obj('fwd_model', params);

    % code from compare_2d_algs example in EIDORS doc
    inv2d= eidors_obj('inv_model', 'EIT inverse');
    inv2d.reconst_type= 'difference';
    inv2d.jacobian_bkgnd.value= 1;
    inv2d.fwd_model= mdl_2d;
    %inv2d.fwd_model.np_fwd_solve.perm_sym= '{y}';
    inv2d.parameters.term_tolerance= 1e-4;

    %inv2d.fwd_model = mdl_normalize(inv2d.fwd_model,1);

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
        inv2d.hyperparameter.noise_figure= 2;
        inv2d.hyperparameter.tgt_elems= 1:4;
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

    inv2d.name= 'dsh_rect'; 
    inv2d= eidors_obj('inv_model', inv2d);
    valid_inv_model(inv2d);
    [stim, meas_select] = mk_stim_patterns(32,1,'{op}','{ad}',options,1);
    inv2d.fwd_model.stimulation = stim;
    inv2d.fwd_model.meas_select = meas_select;
    img = inv_solve(inv2d, homg, objs);
    show_slices(img);

end

function [data_objs, data_homg] = getinputs(logtimes, presstimes, responses)
    smoothval = 5;

    presstimes = [presstimes logtimes(end)];
    pressinds = zeros([6, 1]);
    
    for i = 1:5
        pressinds(i) = find(logtimes>presstimes(i), 1) - 2;
    end
    pressinds(6) = size(responses, 1) - smoothval;

    data_homg = zeros([1024, 1]);
    data_objs = zeros([1024, 1]);


    for i = 1:1024
        smoothedresponse = smooth(responses(:, i), smoothval);
        magnitudes = zeros([5, 1]);
        for j = 1:5
            region = smoothedresponse(pressinds(j):pressinds(j+1));
            magnitudes(j) = max(region) - min(region);
        end
        data_objs(i) = mean(magnitudes);
    end
end