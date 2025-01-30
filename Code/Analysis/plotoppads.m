clear
load("topChannels_abscorr.mat");
%plotElectrodeChannels('896',sorted_indices ,169)
%plotTopChannelsCirclesPairs('896',sorted_indices ,25)
%plotElectrodeUsage('896', sorted_indices,25)
plotTopChannelsWithSummary('896', sorted_indices,25)
%plotTopChannelsSummary('896', sorted_indices,640)

function plotElectrodeChannels(electrodeType, channel_list, no_channels)
    % Load the data
    load("oppad32.mat");
    load("electrodeposition.mat")

    % % Set dimensions and gap
    % height = 30; % height in mm
    % width = 24; % width in mm
    % gap = 3; % gap between electrodes in mm
    % 
    % % Calculate the number of electrodes
    % num_vert = height / gap + 1; % +1 to include corners
    % num_horiz = width / gap + 1;
    % 
    % % Generate electrode positions
    % left_side = [zeros(num_vert, 1), (0:gap:gap*(num_vert-1))'];           % Left side
    % right_side = [width * ones(num_vert, 1), (0:gap:gap*(num_vert-1))'];   % Right side
    % bottom_side = [(0:gap:gap*(num_horiz-1))', zeros(num_horiz, 1)];       % Bottom side
    % top_side = [(0:gap:gap*(num_horiz-1))', height * ones(num_horiz, 1)];  % Top side
    % electrodepositions = unique([left_side; right_side; bottom_side; top_side], 'rows');

    % Determine number of channels based on electrodeType
    if strcmp(electrodeType, '896')
        totalChannels = 896;
        loopIndices = channel_list(1:no_channels); % Change this range for different visualizations
    elseif strcmp(electrodeType, '1024')
        totalChannels = 1024;
        loopIndices = 1:length(oppads_withzeros); % Adjust range for 1024 logic
    else
        error('Invalid electrode type. Use "896" or "1024".');
    end

    % Set up the figure for plotting
    figure;
    hold on;
    scatter(electrodepositions(:, 1), electrodepositions(:, 2), 30, "k", "filled", 'DisplayName', 'Electrodes');
    title(sprintf('Electrode Frame and Channels (%s Electrodes)', electrodeType), 'FontSize', 14);
    xlabel('X Position (mm)', 'FontSize', 12);
    ylabel('Y Position (mm)', 'FontSize', 12);
    grid on;
    axis equal;
    xlim([-2, 35 + 2]);
    ylim([-2, 30 + 2]);

    % Flags to add legend entries only once
    injectionPathPlotted = false;
    measurementPathPlotted = false;
    c = 0;
    % Plot the channels
    for i = loopIndices
        c =c+1;
        % Plot injection electrodes and path
        if ~injectionPathPlotted
            plot(electrodepositions(oppads(i, 1:2), 1), electrodepositions(oppads(i, 1:2), 2), ...
                "r-", "LineWidth", 2, 'DisplayName', 'Injection Path');
            injectionPathPlotted = true;
        else
            plot(electrodepositions(oppads(i, 1:2), 1), electrodepositions(oppads(i, 1:2), 2), ...
                "r-", "LineWidth", 2, 'HandleVisibility', 'off');
        end
        scatter(electrodepositions(oppads(i, 1:2), 1), electrodepositions(oppads(i, 1:2), 2), ...
            50, "r", "filled", 'HandleVisibility', 'off');

        % Plot measurement electrodes and path
        % if ~measurementPathPlotted
        %     plot(electrodepositions(oppads(i, 3:4), 1), electrodepositions(oppads(i, 3:4), 2), ...
        %         "b-", "LineWidth", 2, 'DisplayName', 'Measurement Path');
        %     measurementPathPlotted = true;
        % else
        %     plot(electrodepositions(oppads(i, 3:4), 1), electrodepositions(oppads(i, 3:4), 2), ...
        %         "b-", "LineWidth", 2, 'HandleVisibility', 'off');
        % end
        % scatter(electrodepositions(oppads(i, 3:4), 1), electrodepositions(oppads(i, 3:4), 2), ...
        %     50, "b", "filled", 'HandleVisibility', 'off');

        % Update title with current channel
        title(sprintf('Plotted %d of %d', c, no_channels), 'FontSize', 14);
        % pause(0.1); % Pause for visualization
    end

    % Finalize the plot
    legend('Location', 'northeastoutside');
    hold off;
 end

function plotTopChannelsCirclesPairs(electrodeType, channel_list, no_channels)
    % Load the data
    load("oppad32.mat");
    load("electrodeposition.mat");

    % Determine number of channels based on electrodeType
    if strcmp(electrodeType, '896')
        totalChannels = 896;
        loopIndices = channel_list(1:no_channels); % Top N channels
    elseif strcmp(electrodeType, '1024')
        totalChannels = 1024;
        loopIndices = 1:length(oppads_withzeros); % Adjust range for 1024 logic
    else
        error('Invalid electrode type. Use "896" or "1024".');
    end

    % Set up figures for injection and read pairings
    figure_inject = figure;
    figure_read = figure;

    % Determine subplot grid layout
    numRows = ceil(sqrt(no_channels));
    numCols = ceil(no_channels / numRows);

    % Loop over the top channels and plot for both injection and read pairs
    for i = 1:no_channels
        channel_idx = loopIndices(i);

        % --- Injection Pairing Plot ---
        figure(figure_inject);
        subplot(numRows, numCols, i);
        hold on;
        scatter(electrodepositions(:, 1), electrodepositions(:, 2), 30, "k", "filled", 'DisplayName', 'Electrodes');
        scatter(electrodepositions(oppads(channel_idx, 1:2), 1), ...
                electrodepositions(oppads(channel_idx, 1:2), 2), ...
                100, "r", "filled", 'DisplayName', 'Injection Pair');
        title(sprintf('Injection Channel %d', channel_idx), 'FontSize', 10);
        xlabel('X (mm)', 'FontSize', 8);
        ylabel('Y (mm)', 'FontSize', 8);
        grid on;
        axis equal;
        xlim([-2, 35 + 2]);
        ylim([-2, 30 + 2]);
        hold off;

        % --- Read Pairing Plot ---
        figure(figure_read);
        subplot(numRows, numCols, i);
        hold on;
        scatter(electrodepositions(:, 1), electrodepositions(:, 2), 30, "k", "filled", 'DisplayName', 'Electrodes');
        scatter(electrodepositions(oppads(channel_idx, 3:4), 1), ...
                electrodepositions(oppads(channel_idx, 3:4), 2), ...
                100, "b", "filled", 'DisplayName', 'Read Pair');
        title(sprintf('Read Channel %d', channel_idx), 'FontSize', 10);
        xlabel('X (mm)', 'FontSize', 8);
        ylabel('Y (mm)', 'FontSize', 8);
        grid on;
        axis equal;
        xlim([-2, 35 + 2]);
        ylim([-2, 30 + 2]);
        hold off;
    end

    % Finalize figures with super titles
    figure(figure_inject);
    sgtitle(sprintf('Top %d Channels - Injection Pairings', no_channels), 'FontSize', 14);

    figure(figure_read);
    sgtitle(sprintf('Top %d Channels - Read Pairings', no_channels), 'FontSize', 14);
end

function plotElectrodeUsage(electrodeType, channel_list, no_channels)
    % Load the data
    load("oppad32.mat");
    load("electrodeposition.mat");

    % Determine the number of channels based on electrodeType
    if strcmp(electrodeType, '896')
        totalChannels = 896;
        loopIndices = channel_list(1:no_channels); % Top N channels
    elseif strcmp(electrodeType, '1024')
        totalChannels = 1024;
        loopIndices = 1:length(oppads_withzeros); % Adjust range for 1024 logic
    else
        error('Invalid electrode type. Use "896" or "1024".');
    end

    % Initialize frequency counters for electrodes
    num_electrodes = size(electrodepositions, 1); % Total electrodes
    inject_usage = zeros(num_electrodes, 1); % Frequency of injection electrodes
    read_usage = zeros(num_electrodes, 1); % Frequency of read electrodes

    % Count the usage of each electrode in injection and read pairings
    for i = loopIndices
        inject_usage(oppads(i, 1)) = inject_usage(oppads(i, 1)) + 1; % Increment for first inject electrode
        inject_usage(oppads(i, 2)) = inject_usage(oppads(i, 2)) + 1; % Increment for second inject electrode
        read_usage(oppads(i, 3)) = read_usage(oppads(i, 3)) + 1; % Increment for first read electrode
        read_usage(oppads(i, 4)) = read_usage(oppads(i, 4)) + 1; % Increment for second read electrode
    end

    % --- Plot Heatmaps ---
    % Injection usage heatmap
    figure;
    scatter(electrodepositions(:, 1), electrodepositions(:, 2), 200, inject_usage, "filled");
    colorbar;
    colormap('hot');
    title(sprintf('Injection Electrode Usage Heatmap (Top %d Channels)', no_channels), 'FontSize', 14);
    xlabel('X Position (mm)', 'FontSize', 12);
    ylabel('Y Position (mm)', 'FontSize', 12);
    axis equal;
    grid on;

    % Read usage heatmap
    figure;
    scatter(electrodepositions(:, 1), electrodepositions(:, 2), 200, read_usage, "filled");
    colorbar;
    colormap('winter');
    title(sprintf('Read Electrode Usage Heatmap (Top %d Channels)', no_channels), 'FontSize', 14);
    xlabel('X Position (mm)', 'FontSize', 12);
    ylabel('Y Position (mm)', 'FontSize', 12);
    axis equal;
    grid on;
end

function plotTopChannelsVectors(electrodeType, channel_list, no_channels)
    % Load the data
    load("oppad32.mat");
    load("electrodeposition.mat");

    % Determine number of channels based on electrodeType
    if strcmp(electrodeType, '896')
        totalChannels = 896;
        loopIndices = channel_list(1:no_channels); % Top N channels
    elseif strcmp(electrodeType, '1024')
        totalChannels = 1024;
        loopIndices = 1:length(oppads_withzeros); % Adjust range for 1024 logic
    else
        error('Invalid electrode type. Use "896" or "1024".');
    end

    % Set up the figure for subplots
    figure;
    numRows = ceil(sqrt(no_channels));
    numCols = ceil(no_channels / numRows);

    % Loop over the top channels and plot each in a subplot
    for i = 1:no_channels
        subplot(numRows, numCols, i);
        channel_idx = loopIndices(i);

        % Get electrode positions for the pair
        p1 = electrodepositions(oppads(channel_idx, 1), :); % Start of the pair
        p2 = electrodepositions(oppads(channel_idx, 2), :); % End of the pair

        % Compute midpoint and vector components
        midpoint = (p1 + p2) / 2;
        vector = p2 - p1;

        % Plot all electrodes
        hold on;
        scatter(electrodepositions(:, 1), electrodepositions(:, 2), 30, "k", "filled", 'DisplayName', 'Electrodes');

        % Highlight the electrode pair
        scatter([p1(1), p2(1)], [p1(2), p2(2)], 100, "r", "filled", 'DisplayName', 'Channel Electrodes');

        % Plot vector with arrow
        quiver(midpoint(1), midpoint(2), vector(1), vector(2), 0, ...
            'Color', 'b', 'LineWidth', 1.5, 'MaxHeadSize', 0.5, 'DisplayName', 'Direction Vector');

        % Configure the plot appearance
        title(sprintf('Channel %d', channel_idx), 'FontSize', 10);
        xlabel('X (mm)', 'FontSize', 8);
        ylabel('Y (mm)', 'FontSize', 8);
        grid on;
        axis equal;
        xlim([-2, 35 + 2]);
        ylim([-2, 30 + 2]);
        hold off;
    end

    % Finalize the figure
    sgtitle(sprintf('Top %d Channels - Vectors', no_channels), 'FontSize', 14); % Super-title
end

function plotTopChannelsWithSummary(electrodeType, channel_list, no_channels)
    % Load the data
    load("oppad32.mat");
    load("electrodeposition.mat");

    % Determine number of channels based on electrodeType
    if strcmp(electrodeType, '896')
        totalChannels = 896;
        loopIndices = channel_list(1:no_channels); % Top N channels
    elseif strcmp(electrodeType, '1024')
        totalChannels = 1024;
        loopIndices = 1:length(oppads_withzeros); % Adjust range for 1024 logic
    else
        error('Invalid electrode type. Use "896" or "1024".');
    end

    % Initialize arrays for aggregate summary
    injection_midpoints = [];
    injection_vectors = [];
    read_midpoints = [];
    read_vectors = [];

    % Set up the figure for subplots
    figure;
    numRows = ceil(sqrt(no_channels));
    numCols = ceil(no_channels / numRows);

    % Loop over the top channels and plot each in a subplot
    for i = 1:no_channels
        subplot(numRows, numCols, i);
        channel_idx = loopIndices(i);

        % Get electrode positions for the injection pair
        p1_inj = electrodepositions(oppads(channel_idx, 1), :); % Start of the injection pair
        p2_inj = electrodepositions(oppads(channel_idx, 2), :); % End of the injection pair
        midpoint_inj = (p1_inj + p2_inj) / 2; % Midpoint
        vector_inj = p2_inj - p1_inj; % Vector from p1 to p2

        % Store for aggregate summary
        injection_midpoints = [injection_midpoints; midpoint_inj];
        injection_vectors = [injection_vectors; vector_inj];

        % Plot the injection pair
        hold on;
        scatter(electrodepositions(:, 1), electrodepositions(:, 2), 30, "k", "filled", 'DisplayName', 'Electrodes');
        scatter([p1_inj(1), p2_inj(1)], [p1_inj(2), p2_inj(2)], 100, "r", "filled", 'DisplayName', 'Injection Pair');
        quiver(midpoint_inj(1), midpoint_inj(2), vector_inj(1), vector_inj(2), 0, ...
            'Color', 'r', 'LineWidth', 1.5, 'MaxHeadSize', 0.5, 'DisplayName', 'Injection Vector');

        % Get electrode positions for the read pair
        p1_read = electrodepositions(oppads(channel_idx, 3), :); % Start of the read pair
        p2_read = electrodepositions(oppads(channel_idx, 4), :); % End of the read pair
        midpoint_read = (p1_read + p2_read) / 2; % Midpoint
        vector_read = p2_read - p1_read; % Vector from p1 to p2

        % Store for aggregate summary
        read_midpoints = [read_midpoints; midpoint_read];
        read_vectors = [read_vectors; vector_read];

        % Plot the read pair
        scatter([p1_read(1), p2_read(1)], [p1_read(2), p2_read(2)], 100, "b", "filled", 'DisplayName', 'Read Pair');
        quiver(midpoint_read(1), midpoint_read(2), vector_read(1), vector_read(2), 0, ...
            'Color', 'b', 'LineWidth', 1.5, 'MaxHeadSize', 0.5, 'DisplayName', 'Read Vector');

        % Configure the plot appearance
        title(sprintf('Channel %d', channel_idx), 'FontSize', 10);
        xlabel('X (mm)', 'FontSize', 8);
        ylabel('Y (mm)', 'FontSize', 8);
        grid on;
        axis equal;
        xlim([-2, 35 + 2]);
        ylim([-2, 30 + 2]);
        hold off;
    end

    % Plot the aggregate summary of injection vectors
    figure;
    hold on;
    scatter(electrodepositions(:, 1), electrodepositions(:, 2), 30, "k", "filled", 'DisplayName', 'Electrodes');
    quiver(injection_midpoints(:, 1), injection_midpoints(:, 2), injection_vectors(:, 1), injection_vectors(:, 2), 0, ...
        'Color', 'r', 'LineWidth', 1, 'MaxHeadSize', 0.5, 'DisplayName', 'Injection Vectors');

    % Plot the average injection vector
    avg_vector_inj = mean(injection_vectors, 1);
    avg_midpoint_inj = mean(injection_midpoints, 1);
    quiver(avg_midpoint_inj(1), avg_midpoint_inj(2), avg_vector_inj(1), avg_vector_inj(2), 0, ...
        'Color', 'm', 'LineWidth', 2, 'MaxHeadSize', 1, 'DisplayName', 'Avg Injection Vector');

    % Configure the aggregate plot for injection vectors
    title('Aggregate Summary of Injection Vectors', 'FontSize', 14);
    xlabel('X Position (mm)', 'FontSize', 12);
    ylabel('Y Position (mm)', 'FontSize', 12);
    legend('Location', 'northeastoutside');
    grid on;
    axis equal;
    xlim([-2, 35 + 2]);
    ylim([-2, 30 + 2]);
    hold off;

    % Plot the aggregate summary of read vectors
    figure;
    hold on;
    scatter(electrodepositions(:, 1), electrodepositions(:, 2), 30, "k", "filled", 'DisplayName', 'Electrodes');
    quiver(read_midpoints(:, 1), read_midpoints(:, 2), read_vectors(:, 1), read_vectors(:, 2), 0, ...
        'Color', 'b', 'LineWidth', 1, 'MaxHeadSize', 0.5, 'DisplayName', 'Read Vectors');

    % Plot the average read vector
    avg_vector_read = mean(read_vectors, 1);
    avg_midpoint_read = mean(read_midpoints, 1);
    quiver(avg_midpoint_read(1), avg_midpoint_read(2), avg_vector_read(1), avg_vector_read(2), 0, ...
        'Color', 'c', 'LineWidth', 2, 'MaxHeadSize', 1, 'DisplayName', 'Avg Read Vector');

    % Configure the aggregate plot for read vectors
    title('Aggregate Summary of Read Vectors', 'FontSize', 14);
    xlabel('X Position (mm)', 'FontSize', 12);
    ylabel('Y Position (mm)', 'FontSize', 12);
    legend('Location', 'northeastoutside');
    grid on;
    axis equal;
    xlim([-2, 35 + 2]);
    ylim([-2, 30 + 2]);
    hold off;

    figure()
    hold on;
    % Plot the average injection vector
    avg_vector_inj = mean(injection_vectors, 1);
    avg_vector_inj = avg_vector_inj / norm(avg_vector_inj) * 10; % Normalize and scale
    avg_midpoint_inj = mean(injection_midpoints, 1);
    quiver(avg_midpoint_inj(1), avg_midpoint_inj(2), avg_vector_inj(1), avg_vector_inj(2), 0, ...
        'Color', 'r', 'LineWidth', 2, 'MaxHeadSize', 1, 'DisplayName', 'Avg Injection Vector');

    % Plot the average read vector
    avg_vector_read = mean(read_vectors, 1);
    avg_vector_read = avg_vector_read / norm(avg_vector_read) * 10; % Normalize and scale
    avg_midpoint_read = mean(read_midpoints, 1);
    quiver(avg_midpoint_read(1), avg_midpoint_read(2), avg_vector_read(1), avg_vector_read(2), 0, ...
        'Color', 'b', 'LineWidth', 2, 'MaxHeadSize', 1, 'DisplayName', 'Avg Read Vector');    

    legend('AutoUpdate','off');
    % Add a centered axis
    line([0, 36], [14, 14], 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1.5 ); % X-axis
    line([18, 18], [0, 28], 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1.5); % Y-axis
    

    scatter(electrodepositions(:, 1), electrodepositions(:, 2), 30, "k", "filled", 'DisplayName', 'Electrodes');
    hold off;
    
    figure()
    hold on;
    
    % Plot the normalized average vectors starting from (0,0)
    quiver(0, 0, avg_vector_inj(1), avg_vector_inj(2), 0, ...
        'Color', 'r', 'LineWidth', 2, 'MaxHeadSize', 2, 'DisplayName', 'Normalized Avg Injection Vector');
    quiver(0, 0, avg_vector_read(1), avg_vector_read(2), 0, ...
        'Color', 'b', 'LineWidth', 2, 'MaxHeadSize', 2, 'DisplayName', 'Normalized Avg Read Vector');
    legend('AutoUpdate','off');
    % Add a centered axis
    line([-12, 12], [0, 0], 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1.5 ); % X-axis
    line([0, 0], [-12, 12], 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1.5); % Y-axis
    
    % Configure plot appearance
    title('Normalized Average Vectors (Centered)');
    xlabel('X Axis');
    ylabel('Y Axis');
    legend('Location', 'northeastoutside');
    grid on;
    axis equal; % Ensure equal scaling for X and Y axes
    xlim([-12, 12]); % Define axis limits
    ylim([-12, 12]);
    hold off;
end

function plotTopChannelsSummary(electrodeType, channel_list, no_channels)
    % Load the data
    load("oppad32.mat");
    load("electrodeposition.mat");

    % Determine number of channels based on electrodeType
    if strcmp(electrodeType, '896')
        totalChannels = 896;
        loopIndices = channel_list(1:no_channels); % Top N channels
    elseif strcmp(electrodeType, '1024')
        totalChannels = 1024;
        loopIndices = 1:length(oppads_withzeros); % Adjust range for 1024 logic
    else
        error('Invalid electrode type. Use "896" or "1024".');
    end

    % Initialize arrays for aggregate summary
    injection_midpoints = [];
    injection_vectors = [];
    read_midpoints = [];
    read_vectors = [];

    % Set up the figure for subplots
    numRows = ceil(sqrt(no_channels));
    numCols = ceil(no_channels / numRows);

    % Loop over the top channels and plot each in a subplot
    for i = 1:no_channels
        channel_idx = loopIndices(i);

        % Get electrode positions for the injection pair
        p1_inj = electrodepositions(oppads(channel_idx, 1), :); % Start of the injection pair
        p2_inj = electrodepositions(oppads(channel_idx, 2), :); % End of the injection pair
        midpoint_inj = (p1_inj + p2_inj) / 2; % Midpoint
        vector_inj = p2_inj - p1_inj; % Vector from p1 to p2

        % Store for aggregate summary
        injection_midpoints = [injection_midpoints; midpoint_inj];
        injection_vectors = [injection_vectors; vector_inj];

        % Get electrode positions for the read pair
        p1_read = electrodepositions(oppads(channel_idx, 3), :); % Start of the read pair
        p2_read = electrodepositions(oppads(channel_idx, 4), :); % End of the read pair
        midpoint_read = (p1_read + p2_read) / 2; % Midpoint
        vector_read = p2_read - p1_read; % Vector from p1 to p2

        % Store for aggregate summary
        read_midpoints = [read_midpoints; midpoint_read];
        read_vectors = [read_vectors; vector_read];
    end

    % Plot the aggregate summary of injection vectors
    figure;
    hold on;
    scatter(electrodepositions(:, 1), electrodepositions(:, 2), 30, "k", "filled", 'DisplayName', 'Electrodes');
    quiver(injection_midpoints(:, 1), injection_midpoints(:, 2), injection_vectors(:, 1), injection_vectors(:, 2), 0, ...
        'Color', 'r', 'LineWidth', 1, 'MaxHeadSize', 0.5, 'DisplayName', 'Injection Vectors');

    % Plot the average injection vector
    avg_vector_inj = mean(injection_vectors, 1);
    avg_midpoint_inj = mean(injection_midpoints, 1);
    quiver(avg_midpoint_inj(1), avg_midpoint_inj(2), avg_vector_inj(1), avg_vector_inj(2), 0, ...
        'Color', 'm', 'LineWidth', 2, 'MaxHeadSize', 1, 'DisplayName', 'Avg Injection Vector');


    % Configure the aggregate plot for read vectors
    title('Aggregate Summary of Read Vectors', 'FontSize', 14);
    xlabel('X Position (mm)', 'FontSize', 12);
    ylabel('Y Position (mm)', 'FontSize', 12);
    legend('Location', 'northeastoutside');
    grid on;
    axis equal;
    xlim([0, 38]);
    ylim([0, 28]);
    hold off;

    figure()
    hold on;
    % Plot the average injection vector
    avg_vector_inj = mean(injection_vectors, 1);
    avg_vector_inj = avg_vector_inj / norm(avg_vector_inj) * 10; % Normalize and scale
    avg_midpoint_inj = mean(injection_midpoints, 1);
    quiver(avg_midpoint_inj(1), avg_midpoint_inj(2), avg_vector_inj(1), avg_vector_inj(2), 0, ...
        'Color', 'r', 'LineWidth', 2, 'MaxHeadSize', 1, 'DisplayName', 'Avg Injection Vector');

    % Plot the average read vector
    avg_vector_read = mean(read_vectors, 1);
    avg_vector_read = avg_vector_read / norm(avg_vector_read) * 10; % Normalize and scale
    avg_midpoint_read = mean(read_midpoints, 1);
    quiver(avg_midpoint_read(1), avg_midpoint_read(2), avg_vector_read(1), avg_vector_read(2), 0, ...
        'Color', 'b', 'LineWidth', 2, 'MaxHeadSize', 1, 'DisplayName', 'Avg Read Vector');    
    
    % Add the screwdriver
    screwdriver_x = [0, 36]; % Screwdriver aligned perpendicular to the axis
    screwdriver_y = [14, 14]; % Length of the screwdriver
    plot(screwdriver_x, screwdriver_y, 'k','LineStyle','--','LineWidth', 2, 'DisplayName', 'Screwdriver'); % Red line for screwdriver
    
    

    % Indicate the twisting motion with an arc
    theta = linspace(-pi/4, pi/4, 50); % Arc from -45 to +45 degrees
    arc_radius = 1.2; % Radius of the arc
    arc_x = arc_radius * cos(theta); % X-coordinates of the arc
    arc_y = arc_radius * sin(theta); % Y-coordinates of the arc
    plot(4+arc_x, 14+arc_y, 'g', 'LineWidth', 1.5, 'DisplayName', 'Rotation'); % Green arc for rotation
    plot(28+arc_x, 14+arc_y, 'g', 'LineWidth', 1.5, 'DisplayName', 'Rotation'); % Green arc for rotation
    
    scatter(18,14,'o','filled','k','DisplayName','Centre','LineWidth',2 )
    legend('AutoUpdate','off')
    
    line([0, 0], [28, 0], 'Color', 'k', 'LineWidth', 1.5 );
    line([36, 36], [28, 0], 'Color', 'k', 'LineWidth', 1.5 );
    line([0, 36], [28, 28], 'Color', 'k', 'LineWidth', 1.5); 
    line([0, 36], [0, 0], 'Color', 'k', 'LineWidth', 1.5);

    plot(screwdriver_x, screwdriver_y, 'k','LineStyle','--','LineWidth', 2, 'DisplayName', 'Screwdriver');
    % Configure the aggregate plot for injection vectors
    title('Aggregate Summary of Injection Vectors', 'FontSize', 14);
    xlabel('X Position (mm)', 'FontSize', 12);
    ylabel('Y Position (mm)', 'FontSize', 12);
    legend('Location', 'northeastoutside');
    grid on;
    axis equal;
    xlim([0, 36]);
    ylim([0, 28]);
    hold off;

    % Plot the aggregate summary of read vectors
    figure;
    hold on;

    scatter(electrodepositions(:, 1), electrodepositions(:, 2), 30, "k", "filled", 'DisplayName', 'Electrodes');
    quiver(read_midpoints(:, 1), read_midpoints(:, 2), read_vectors(:, 1), read_vectors(:, 2), 0, ...
        'Color', 'b', 'LineWidth', 1, 'MaxHeadSize', 0.5, 'DisplayName', 'Read Vectors');

    % Plot the average read vector
    avg_vector_read = mean(read_vectors, 1);
    avg_midpoint_read = mean(read_midpoints, 1);
    quiver(avg_midpoint_read(1), avg_midpoint_read(2), avg_vector_read(1), avg_vector_read(2), 0, ...
        'Color', 'c', 'LineWidth', 2, 'MaxHeadSize', 1, 'DisplayName', 'Avg Read Vector');

    legend('AutoUpdate','off');
    % Add a centered axis
    line([0, 36], [14, 14], 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1.5 ); % X-axis
    line([18, 18], [0, 28], 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1.5); % Y-axis
    ylim([0,28]); 
    xlim([0,36]);
    scatter(electrodepositions(:, 1), electrodepositions(:, 2), 30, "k", "filled", 'DisplayName', 'Electrodes');
    hold off;
    
    figure()
    hold on;
    
    % Plot the normalized average vectors starting from (0,0)
    quiver(0, 0, avg_vector_inj(1), avg_vector_inj(2), 0, ...
        'Color', 'r', 'LineWidth', 2, 'MaxHeadSize', 2, 'DisplayName', 'Normalized Avg Injection Vector');
    quiver(0, 0, avg_vector_read(1), avg_vector_read(2), 0, ...
        'Color', 'b', 'LineWidth', 2, 'MaxHeadSize', 2, 'DisplayName', 'Normalized Avg Read Vector');
    legend('AutoUpdate','off');
    % Add a centered axis
    line([-12, 12], [0, 0], 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1.5 ); % X-axis
    line([0, 0], [-12, 12], 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1.5); % Y-axis
    
    % Configure plot appearance
    title('Normalized Average Vectors (Centered)');
    xlabel('X Axis');
    ylabel('Y Axis');
    legend('Location', 'northeastoutside');
    grid on;
    axis equal; % Ensure equal scaling for X and Y axes
    xlim([-12, 12]); % Define axis limits
    ylim([-12, 12]);
    hold off;
end
