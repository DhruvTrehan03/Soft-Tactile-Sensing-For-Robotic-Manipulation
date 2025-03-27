% Define the figure file names
fig_files = {'10mm Sigma K.fig', '20mm Sigma K.fig', '30mm Sigma K.fig', '40mm Sigma K.fig'};

% Initialize cell arrays to store the heatmap data
heatmap_data = cell(1, 4);
XData = cell(1, 4);
YData = cell(1, 4);

% Load figures and extract heatmap data
for i = 1:4
    fig = openfig(fig_files{i}); % Open figure dvisibly
    ax = findall(fig, 'Type', 'axes');  % Get axes handle
    img = findall(ax, 'Type', 'Image'); % Find heatmap image
    heatmap_data{i} = img.CData;        % Extract heatmap matrix
    XData{i} = img.XData;               % Extract X axis data
    YData{i} = img.YData;               % Extract Y axis data
end

% Compute the difference heatmaps (subtract first heatmap from others)
diff_maps = cell(1, 3);
for i = 1:4
    diff_maps{i} = heatmap_data{i} - heatmap_data{1}; % Subtract first heatmap
end

% Plot the difference heatmaps

for i = 1:4
    figure;
    imagesc(XData{1}, YData{1}, diff_maps{i}); % Plot heatmap difference
    colorbar;
    xlabel('k Value');
    ylabel('Sigma');
    title(sprintf('Difference Heatmap %d - 1', i));
    axis xy;
    colormap hot; % Use jet colormap for better visualization
    set(gca, 'YDir', 'normal'); % Ensure correct Y-axis orientation
end
