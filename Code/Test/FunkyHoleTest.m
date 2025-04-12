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


%%
% MultiSelectFEM_FunkyPerms();
% compareVoltageSimilarities();
% plotSimilarityMatrixByShape();
%plotNormalizedSimilarityMatrixByShape();
simulateConfusionMatrix('FunkyPerms',1000,0.00002);

%%
function TwoD()
    clear
    run('Source/eidors-v3.11-ng/eidors/eidors_startup.m'); % Initialize EIDORS
    
    %% Define the main rectangle (outer boundary)
    width  = 4.4;
    height = 3.6;
    outer_xy = [0,  0;
                width,  0;
                width, height;
                0,  height];  % Counterclockwise order5090
    
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
    % select_fcn = @(x,y,z)((x-2.2).^2+(y-1.5).^2<r^2);
    % select_fcn = @(x,y,z)(x>1.9) & (x<2.5) & (y<1.8) & (y>1.2);
    % select_fcn = @(x,y,z) ( ...
    %     (( (x - 2.2) - (y - 1.5) ) / sqrt(2) > (1.9 - 2.2)) & ...
    %     (( (x - 2.2) - (y - 1.5) ) / sqrt(2) < (2.5 - 2.2)) & ...
    %     (( (x - 2.2) + (y - 1.5) ) / sqrt(2) > (1.2 - 1.5)) & ...
    %     (( (x - 2.2) + (y - 1.5) ) / sqrt(2) < (1.8 - 1.5)) ...
    % );
    x_c= 2.2;
    y_c = 1.4;
    s=0.8;
    select_fcn = @(x, y, z) inpolygon(x, y, ...
    [x_c, x_c - s/2, x_c + s/2], ...
    [y_c + sqrt(3)/3 * s, y_c - sqrt(3)/6 * s, y_c - sqrt(3)/6 * s]);



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
end

function ThreeD()
    clear
    run('Source/eidors-v3.11-ng/eidors/eidors_startup.m'); % Initialize EIDORS
    
end

function MultiSelectFEM()
    clear
    run('Source/eidors-v3.11-ng/eidors/eidors_startup.m'); % Initialize EIDORS

    % Define model
    width  = 4.4;
    height = 3.6;
    outer_xy = [0,  0;
                width,  0;
                width, height;
                0,  height];  

    hole1_xy = [1,   1;
                1.5, 1;
                1.5, 2;
                1,   2];  

    hole2_xy = [2.9, 1;
                3.4, 1;
                3.4, 2;
                2.9, 2];  

    shape = {outer_xy, hole1_xy, hole2_xy, 0.1};  

    % Electrodes
    elec_outer   = [0, 1];  
    n_elec = 12;
    elec_inner1  = [1.5 * ones(n_elec,1), linspace(1,2,n_elec)'];  
    elec_inner2  = [2.9 * ones(n_elec,1), linspace(1,2,n_elec)'];  
    elec_pos = {elec_outer, elec_inner1, elec_inner2}; 

    % Generate model
    mdl = ng_mk_2d_model(shape, elec_pos);
    stim = mk_stim_patterns(2*n_elec, 1, [0, 1], [1, 0], {'no_meas_current'}, 5);

    % Define Select Functions
    select_fcns = {
        @(x, y, z) (x > 1.9) & (x < 2.5) & (y < 1.8) & (y > 1.2),  % Cube Region
        @(x, y, z) ( ...
            (( (x - 2.2) - (y - 1.5) ) / sqrt(2) > (1.9 - 2.2)) & ...
            (( (x - 2.2) - (y - 1.5) ) / sqrt(2) < (2.5 - 2.2)) & ...
            (( (x - 2.2) + (y - 1.5) ) / sqrt(2) > (1.2 - 1.5)) & ...
            (( (x - 2.2) + (y - 1.5) ) / sqrt(2) < (1.8 - 1.5)) ...
        ),  % Rotated Cube Region
        @(x, y, z) (x > 1.9) & (x < 2.5) & (y < 2.4) & (y > 1.8),  % Cube Region Shifted
        @(x, y, z) inpolygon(x, y, ...
            [2.2, 2.2 - 0.8/2, 2.2 + 0.8/2], ...
            [1.4 + sqrt(3)/3 * 0.8, 1.4 - sqrt(3)/6 * 0.8, 1.4 - sqrt(3)/6 * 0.8])  % Triangle
    };

    % Base FEM Model (same for all cases)
    plain = mk_image(mdl, 10, 'Plain');
    plain.fwd_model.stimulation = stim;
    plain.fwd_solve.get_all_meas = 1;
    plain_data = fwd_solve(plain);
    plain_volts = rmfield(plain, 'elem_data');
    plain_volts.node_data = plain_data.volt(:,1);

    % Create figure
    figure();
    tiledlayout(4,3, 'TileSpacing', 'Compact', 'Padding', 'Compact');

    % Loop through each select function
    for i = 1:length(select_fcns)
        select_fcn = select_fcns{i};

        % Generate FEM model with perturbation
        perturbed = mk_image(mdl, 10, 'Perturbed');
        perturbed.fwd_model.stimulation = stim;
        perturbed.elem_data = 10 + 0.1 * elem_select(mdl, select_fcn);
        perturbed.fwd_solve.get_all_meas = 1;
        perturbed_data = fwd_solve(perturbed);
        perturbed_volts = rmfield(perturbed, 'elem_data');
        perturbed_volts.node_data = perturbed_data.volt(:,1);

        % Subplot 1: Base FEM Model (No perturbation)
        nexttile;
        show_fem(plain);
        title(sprintf('Base FEM (%d)', i));

        % Subplot 2: FEM Model with Perturbation
        nexttile;
        show_fem(perturbed);
        title(sprintf('Perturbed FEM (%d)', i));

        % Subplot 3: Voltage difference
        nexttile;
        perturbed_volts.node_data = perturbed_volts.node_data - plain_volts.node_data;
        show_fem_enhanced(perturbed_volts);
        title('Voltage Difference to Homogeneous');
    end

    sgtitle('Comparison of Different Select Functions');
end
function MultiSelectFEM_FunkyPerms()
    % Clear workspace and initialize EIDORS
    clear; close all;
    run('Source/eidors-v3.11-ng/eidors/eidors_startup.m');

    %% Define geometry and model
    width  = 2.8;
    height = 3;
    outer_xy = [0, 0;
                width, 0;
                width, height;
                0, height];  

    hole1_xy = [0.2, 1;
                0.7, 1;
                0.7, 2;
                0.2, 2];  

    hole2_xy = [2.1, 1;
                2.6, 1;
                2.6, 2;
                2.1, 2];  

    shape = {outer_xy, hole1_xy, hole2_xy, 0.1};  

    % Electrodes
    elec_outer   = [0, 1];  
    n_elec = 12;
    elec_inner1  = [0.7 * ones(n_elec,1), linspace(1,2,n_elec)'];  
    elec_inner2  = [2.1 * ones(n_elec,1), linspace(1,2,n_elec)'];  
    elec_pos = {elec_outer, elec_inner1, elec_inner2}; 

    % Generate model and stimulation pattern
    mdl = ng_mk_2d_model(shape, elec_pos);
    stim = mk_stim_patterns(2*n_elec, 1, [0,1], [1, 0], {'no_meas_current'}, 5);

    %% Define Positions
    % Position 1: ~1/3 up in y, centered in x
    % Position 2: Center of the domain
    % Position 3: ~2/3 up in y, centered in x
    pos{1} = [width/2, height/4];
    pos{2} = [width/2, 2*height/4];
    pos{3} = [width/2, 3*height/4];

    %% Define Shape Parameters
    % Set sizes so that the shapes are roughly comparable.
    Cube_side = 0.6;
    triangle_side = 0.6;
    Sphere_radius = 0.3;
    pill_length = 0.8;
    pill_width = 0.3;
    heart_size = 0.25; % a scaling parameter

    %% Define Base Shape Functions (centered at the origin)
    Cube_fn = @(x,y) (abs(x) <= Cube_side/2) & (abs(y) <= Cube_side/2);

    % Triangle: equilateral triangle with base horizontal.
    triangle_fn = @(x,y) inpolygon(x, y, ...
        [-triangle_side/2, triangle_side/2, 0], ...
        [-triangle_side/2, -triangle_side/2, triangle_side/2]);
    
    Sphere_fn = @(x,y) (x.^2 + y.^2) <= Sphere_radius^2;
    
    % Pill shape: horizontal rectangle with semicircular ends.
    pill_fn = @(x,y) ( ( (abs(x) <= (pill_length - pill_width)/2) & (abs(y) <= pill_width/2) ) | ...
                        ( (x > (pill_length - pill_width)/2) & (((x - (pill_length - pill_width)/2).^2 + y.^2) <= (pill_width/2)^2) ) | ...
                        ( (x < -(pill_length - pill_width)/2) & (((x + (pill_length - pill_width)/2).^2 + y.^2) <= (pill_width/2)^2) ) );
    
    % Heart shape: an approximate heart curve (shifted so that the heart is centered)
    heart_fn = @(x,y) (((x/heart_size).^2 + ((y+0.25*heart_size)/heart_size).^2 - 1).^3 - (x/heart_size).^2 .* ((y+0.25*heart_size)/heart_size).^3) <= 0;

    shapes = {Cube_fn, triangle_fn, Sphere_fn, pill_fn, heart_fn};
    shape_names = {'C', 'T', 'S', 'P', 'H'};

    %% Define Orientations
    % Orientation 1: original (0 degrees); Orientation 2: rotated by 90 degrees.
    orientations = [0, 90]; % in degrees

    %% Create Folder for Saving Results
    outputFolder = 'FunkyPerms';
    if ~exist(outputFolder, 'dir')
        mkdir(outputFolder);
    end


%% Define Orientation Options Per Shape
% Set orientations for each shape by using a cell array,
% in the same order as "shapes" and "shape_names"
% For the Cube (first shape): use [0, 45] degrees.
% For the Sphere (third shape): use [0] only.
% For others, use [0, 90] (for example).
orientations_options = {
    [0, 45], ...       % Cube
    [0, 90], ...       % Triangle
    [0], ...           % Sphere
    [0, 90], ...       % Pill
    [0, 90]            % Heart
};

%% Build Permutations: a cell array of select functions with unique names.
select_fcns = {};
names = {};

for s = 1:length(shapes)
    base_fn = shapes{s};
    % Get the orientation options for this shape
    current_orientations = orientations_options{s};
    
    for p = 1:length(pos)
        center = pos{p};
        for o = 1:length(current_orientations)
            theta = current_orientations(o);
            % Rotation matrix in degrees
            R = [cosd(theta), -sind(theta); sind(theta), cosd(theta)];
            % Create an anonymous function that shifts the coordinates so that the
            % shape is centered at 'center', applies the rotation, and then tests the base function.
            sel = @(x,y,z) base_fn( R(1,1)*(x-center(1)) + R(1,2)*(y-center(2)), ...
                                      R(2,1)*(x-center(1)) + R(2,2)*(y-center(2)) );
            % Define a unique name: e.g. 'Cube_Pos1_Orient1' and so on.
            name_str = sprintf('%s P%d O%d', shape_names{s}, p, o);
            select_fcns{end+1} = sel;
            names{end+1} = name_str;
        end
    end
end


    %% Obtain the Base (Homogeneous) FEM Voltage Data
    plain = mk_image(mdl, 10, 'Plain');
    plain.fwd_model.stimulation = stim;
    plain.fwd_solve.get_all_meas = 1;
    plain_data = fwd_solve(plain);
    plain_volts = rmfield(plain, 'elem_data');
    plain_volts.node_data = plain_data.volt(:,1);

    %% Loop through Each Permutation, Simulate, Plot, and Save Voltage Data
    % Here we use similar plotting as in your original code.
    for i = 1:length(select_fcns)
        sel_fcn = select_fcns{i};
        name_str = names{i};

        % Create perturbed FEM model based on the select function.
        perturbed = mk_image(mdl, 10, 'Perturbed');
        perturbed.fwd_model.stimulation = stim;
        perturbed.elem_data = 10 + 0.1 * elem_select(mdl, sel_fcn);
        perturbed.fwd_solve.get_all_meas = 1;
        perturbed_data = fwd_solve(perturbed);
        perturbed_volts = rmfield(perturbed, 'elem_data');
        perturbed_volts.node_data = perturbed_data.volt(:,1);
        voltage = perturbed_volts.node_data;
        % Compute voltage difference relative to homogeneous case.
        voltage_diff = voltage - plain_volts.node_data; %Voltages at nodes
        % voltage_diff = perturbed_data.meas-plain_data.meas; %electrode voltages
        % Save the voltage data for this permutation in a .mat file.
        save(fullfile(outputFolder, [name_str, '.mat']), 'voltage_diff','perturbed_volts');

        % Optional: Plot and save the figure.
        figure;
        subplot(1,3,1);
        show_fem(plain);
        title('Base FEM');

        subplot(1,3,2);
        show_fem(perturbed);
        title(name_str);

        subplot(1,3,3);
        temp = perturbed_volts;
        temp.node_data = voltage_diff;
        show_fem_enhanced(temp);
        title('Voltage Difference');
        sgtitle(name_str);

        % Save the current figure as a PNG image.
        saveas(gcf, fullfile(outputFolder, [name_str, '.png']));
        exportgraphics(gcf,fullfile(outputFolder, [name_str, '.pdf']),'ContentType', 'Vector')
        close(gcf);

    end
end

function unnormCorrMat = compareVoltageSimilarities()
    % compareVoltageSimilarities loads voltage difference data from the folder
    % 'FunkyPerms', computes an unnormalized correlation (dot product)
    % similarity matrix, and plots both the similarity matrix and a grid
    % of the voltage difference FEM images.
    %
    % It expects each .mat file in 'FunkyPerms' to contain a variable named
    % 'voltage_diff'.
    
    folderName = 'FunkyPerms';
    files = dir(fullfile(folderName, '*.mat'));
    numFiles = length(files);
    
    % Preallocate cell arrays to store voltage differences and file names.
    voltageDiffs = cell(numFiles, 1);
    perturbVolts = cell(numFiles, 1);
    fileNames = cell(numFiles, 1);
    
    % Load voltage_diff data from each file.
    for i = 1:numFiles
        data = load(fullfile(folderName, files(i).name));
        % Ensure voltage_diff is a column vector.
        voltageDiffs{i} = data.voltage_diff(:);
        perturbVolts{i}=data.perturbed_volts(:);
        fileNames{i} = files(i).name;
    end
    
  % Compute the unnormalized similarity matrix (dot products) for later use.
    unnormCorrMat = zeros(numFiles, numFiles);
    for i = 1:numFiles
        for j = 1:numFiles
            unnormCorrMat(i,j) = abs(mean(voltageDiffs{i} - voltageDiffs{j}));
        end
    end
    
    % Create a normalized correlation matrix using cosine similarity.
    
    
    %% Plot the Similarity Matrix
    figure;
    for i = 1:numFiles
        normCorrMat = normalize(unnormCorrMat(i,:));
        subplot(numFiles,1,i);
        imagesc(normCorrMat);
        set(gca,'XTick', 1:numFiles,'YTick', 1:numFiles, 'YTickLabel', fileNames(i));
    end
    
    % colorbar;
    % title('Unnormalized Correlation Similarity Matrix');
    xlabel('Permutation Index');
    % ylabel('Permutation Index');
    % Optionally, label ticks with file names (if not too many).
    set(gca, 'XTick', 1:numFiles, 'XTickLabel', fileNames);
    
    % %% Plot the FEM Voltage Difference Images in a Grid
    % % Determine an approximate Cube grid size.
    % nRows = ceil(sqrt(numFiles));
    % nCols = ceil(numFiles / nRows);
    % 
    % figure;
    % for i = 1:numFiles
    %     figure;
    %     % Attempt to use the show_fem_enhanced function if it exists.
    %     if exist('show_fem_enhanced', 'file') == 2
    %         % Create a temporary structure with the node_data field.
    %         temp = perturbVolts{i};
    %         temp.node_data = voltageDiffs{i};
    %         try
    %             show_fem_enhanced(temp);
    %         catch
    %             % If show_fem_enhanced errors, fallback to imagesc.
    %             imagesc(voltageDiffs{i});
    %         end
    %     else
    %         imagesc(voltageDiffs{i});
    %     end
    %     title(fileNames{i}, 'Interpreter', 'none');
    %     axis off;
    % end

end

function plotSimilarityMatrixByShape()
    % Folder with voltage difference data files
    folderName = 'FunkyPerms';
    files = dir(fullfile(folderName, '*.mat'));
    numFiles = length(files);
    
    % Preallocate cell arrays for voltage data and file names.
    voltageDiffs = cell(numFiles, 1);
    fileNames = cell(numFiles, 1);
    
    % Load voltage_diff data from each file.
    for i = 1:numFiles
        data = load(fullfile(folderName, files(i).name));
        % Ensure voltage_diff is a column vector.
        voltageDiffs{i} = data.voltage_diff(:);
        % Store the file name (without extension)
        [~, name, ~] = fileparts(files(i).name);
        fileNames{i} = name;
    end
    
    % Compute the overall unnormalized similarity matrix (dot products).
    unnormCorrMat = zeros(numFiles, numFiles);
    for i = 1:numFiles
        for j = 1:numFiles
            unnormCorrMat(i,j) = voltageDiffs{i}' * voltageDiffs{j};
        end
    end
    
    % Group files by shape. Assumes file naming convention: 
    % "Shape_PosX_OrientY" (e.g., 'Cube_Pos1_Orient1')
    shapeNames = cell(numFiles, 1);
    for i = 1:numFiles
        tokens = strsplit(fileNames{i}, ' ');
        shapeNames{i} = tokens{1}; % the first token is assumed to be the shape name
    end
    
    % Determine unique shapes.
    uniqueShapes = unique(shapeNames);
    
    % Create one figure per shape that plots the similarity matrix for that shape.
    for s = 1:length(uniqueShapes)
        currentShape = uniqueShapes{s};
        % Find indices belonging to this shape
        idx = find(strcmp(shapeNames, currentShape));
        % Extract the corresponding similarity submatrix.
        simMatShape = unnormCorrMat(idx, idx);
        % Create a new figure.
        figure('Position',[100, 100, 600, 500]);
        imagesc(simMatShape);
        colorbar;
        title(sprintf('Similarity Matrix for %s', currentShape), 'FontSize', 14);
        xlabel('Permutation Index');
        ylabel('Permutation Index');
        % Label ticks with file names (for this shape)
        shapeFileNames = fileNames(idx);
        set(gca, 'XTick', 1:length(idx), 'XTickLabel', shapeFileNames, ...
                 'YTick', 1:length(idx), 'YTickLabel', shapeFileNames);
        xtickangle(45);
    end
end

function plotNormalizedSimilarityMatrixByShape()
    % plotNormalizedSimilarityMatrixByShape
    % Loads voltage difference data from 'FunkyPerms', groups them by shape 
    % (using file name prefix convention: Shape_PosX_OrientY.mat), then computes
    % a normalized (cosine similarity) correlation matrix for each shape group,
    % and plots it.
    
    folderName = 'FunkyPerms';
    files = dir(fullfile(folderName, '*.mat'));
    numFiles = length(files);
    
    % Preallocate cells for voltage data and file names.
    voltageDiffs = cell(numFiles, 1);
    fileNames = cell(numFiles, 1);
    
    % Load the voltage_diff data from each file.
    for i = 1:numFiles
        data = load(fullfile(folderName, files(i).name));
        % Ensure voltage_diff is a column vector.
        voltageDiffs{i} = data.voltage_diff(:);
        % Store the file name (without extension)
        [~, name, ~] = fileparts(files(i).name);
        fileNames{i} = name;
    end
    
    % Compute the unnormalized similarity matrix (dot products) for later use.
    unnormCorrMat = zeros(numFiles, numFiles);
    for i = 1:numFiles
        for j = 1:numFiles
            unnormCorrMat(i,j) = voltageDiffs{i}' * voltageDiffs{j};
        end
    end
    
    % Compute the norm (squared length) of each voltage vector.
    vecNorm = zeros(numFiles,1);
    for i = 1:numFiles
        vecNorm(i) = sqrt(voltageDiffs{i}' * voltageDiffs{i});
    end
    
    % Create a normalized correlation matrix using cosine similarity.
    normCorrMat = zeros(numFiles, numFiles);
    for i = 1:numFiles
        for j = 1:numFiles
            normCorrMat(i,j) = unnormCorrMat(i,j) / (vecNorm(i) * vecNorm(j));
        end
    end
    
    % Group files by shape. 
    % Assumes file naming convention: "Shape_PosX_OrientY" (e.g., 'Cube_Pos1_Orient1')
    shapeNames = cell(numFiles, 1);
    posNames = cell(numFiles,1);
    orNames = cell(numFiles,1);

    for i = 1:numFiles
        tokens = strsplit(fileNames{i}, ' ');
        shapeNames{i} = tokens{1}; % the first token is assumed to be the shape name
        posNames{i} = tokens{2}; % the first token is assumed to be the shape name
        orNames{i} = tokens{3}; % the first token is assumed to be the shape name
    end

    
    % Determine unique shapes.
    uniqueShapes = unique(shapeNames);
    uniquePos = unique(posNames);
    uniqueOr=unique(orNames);
    
    % Plot a normalized similarity matrix for each shape group.
    for s = 1:length(uniquePos)
        currentShape = uniqueShapes{s};
        % Find indices belonging to this shape.
        idx = find(strcmp(shapeNames, currentShape));
        % Extract the corresponding submatrix.
        subMat = normCorrMat(idx, idx);
        
        % Create a new figure and plot.
        figure('Position',[100, 100, 600, 500]);
        imagesc(subMat);
        colorbar;
        title(sprintf('Normalized Similarity Matrix for %s', currentShape), 'FontSize', 14);
        xlabel('Permutation Index');
        ylabel('Permutation Index');
        
        % Label ticks with file names (for this shape).
        shapeFileNames = fileNames(idx);
        set(gca, 'XTick', 1:length(idx), 'XTickLabel', shapeFileNames, ...
                 'YTick', 1:length(idx), 'YTickLabel', shapeFileNames);
        xtickangle(45);
    end

    for s = 1:length(uniquePos)
        currentPos = uniquePos{s};
        % Find indices belonging to this shape.
        idx = find(strcmp(posNames, currentPos));
        % Extract the corresponding submatrix.
        subMat = normCorrMat(idx, idx);
        
        % Create a new figure and plot.
        figure('Position',[100, 100, 600, 500]);
        imagesc(subMat);
        colorbar;
        title(sprintf('Normalized Similarity Matrix for %s', currentPos), 'FontSize', 14);
        xlabel('Permutation Index');
        ylabel('Permutation Index');
        
        % Label ticks with file names (for this shape).
        shapeFileNames = fileNames(idx);
        set(gca, 'XTick', 1:length(idx), 'XTickLabel', shapeFileNames, ...
                 'YTick', 1:length(idx), 'YTickLabel', shapeFileNames);
        xtickangle(45);
    end

    for s = 1:length(uniqueOr)
        currentOr = uniqueOr{s};
        % Find indices belonging to this shape.
        idx = find(strcmp(orNames, currentOr));
        % Extract the corresponding submatrix.
        subMat = normCorrMat(idx, idx);
        
        % Create a new figure and plot.
        figure('Position',[100, 100, 600, 500]);
        imagesc(subMat);
        colorbar;
        title(sprintf('Normalized Similarity Matrix for %s', currentOr), 'FontSize', 14);
        xlabel('Permutation Index');
        ylabel('Permutation Index');
        
        % Label ticks with file names (for this shape).
        shapeFileNames = fileNames(idx);
        set(gca, 'XTick', 1:length(idx), 'XTickLabel', shapeFileNames, ...
                 'YTick', 1:length(idx), 'YTickLabel', shapeFileNames);
        xtickangle(45);
    end

end


function simulateConfusionMatrix(folderPath, numNoisySamples, noiseLevel)
    % Simulate a confusion matrix based on noisy classification
    
    if nargin < 1, folderPath = 'FunkyPerms'; end
    if nargin < 2, numNoisySamples = 100; end
    if nargin < 3, noiseLevel = 0.00001; end
    % Define desired shape order
    shapeOrder = {'C', 'H', 'P', 'S', 'T'};
   
    % Get and reorder files
    files = dir(fullfile(folderPath, '*.mat'));
    fileNames = {files.name};
    
    % Extract shape name from each filename (assumes format Shape_..._.mat)
    shapesInFiles = regexp(fileNames, '^[^ ]+', 'match', 'once');
    
    % Sort files by matching shape order
    [~, sortIdx] = sort(cellfun(@(s) find(strcmp(shapeOrder, s)), shapesInFiles));
    files = files(sortIdx);

    numPerms = numel(files);
    referenceData = cell(numPerms, 1);
    labels = cell(numPerms, 1);

    fprintf('Loading reference data...\n');
    for i = 1:numPerms
        data = load(fullfile(folderPath, files(i).name));
        labels{i} = files(i).name;
        
        referenceData{i} = data.voltage_diff; % Ensure 'voltageDiff' is saved in each .mat file
    end

    % Flatten all voltage diff data to vectors for comparison
    refVectors = cellfun(@(x) x(:), referenceData, 'UniformOutput', false);
    refVectors = cell2mat(refVectors');
    vectorLength = size(refVectors, 1) / numPerms;

    % Confusion matrix
    confusionMat = zeros(numPerms);

    fprintf('Generating noisy samples and classifying...\n');
    for i = 1:numPerms
        refVec = refVectors(:,i);
        for j = 1:numNoisySamples
            noisySample = refVec + noiseLevel * randn(size(refVec));

            % Compute Euclidean distances to all references
            dists = vecnorm(refVectors - noisySample, 2, 1);

            % Find closest match
            [~, predictedIdx] = min(dists);
            confusionMat(i, predictedIdx) = confusionMat(i, predictedIdx) + 1;
        end
    end

    % Normalize rows to get probabilities
    confusionMat = confusionMat ./ sum(confusionMat, 2);

    % Plot the confusion matrix
    figure;
    imagesc(confusionMat);
    colormap('hot');
    colorbar;
    axis equal tight;
    xticks(1:numPerms);
    yticks(1:numPerms);
    labels = erase({files.name}, '.mat');
    xticklabels(labels);
    yticklabels(labels);
    xtickangle(90);
    xlabel('Predicted');
    ylabel('True');
    title(sprintf('Simulated Confusion Matrix (%d noisy samples each)', numNoisySamples));

    real_conf = csvread("C:\Users\dhruv\Downloads\Temp\Conf_mat.csv");
      % Plot the confusion matrix
    figure;
    imagesc(real_conf(1:23,1:23));
    colormap('hot');
    colorbar;
    axis equal tight;
    xticks(1:numPerms);
    yticks(1:numPerms);
    labels = erase({files.name}, '.mat');
    xticklabels(labels);
    yticklabels(labels);
    xtickangle(90);
    xlabel('Predicted');
    ylabel('True');
    title(sprintf('Real Confusion Matrix (33 samples each)'));

end
