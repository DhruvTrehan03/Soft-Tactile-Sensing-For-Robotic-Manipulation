% filepath: c:\Users\dhruv\Documents\Tripos\SoftTactileSensingForRoboticManipulation\process_3d_scatter.m

% Ensure a .fig file is open
fig = gcf; % Get the current figure
if isempty(fig)
    error('No figure is currently open.');
end

% Extract data from the scatter plot
ax = gca; % Get the current axes
scatterData = findobj(ax, 'Type', 'Scatter');
if isempty(scatterData)
    error('No scatter plot found in the current figure.');
end

% Extract k, sigma, and torque data
xData = scatterData.XData; % k values
yData = scatterData.YData; % sigma values
zData = scatterData.ZData; % torque values

% Perform k-means clustering on torque (zData)
numClusters = 10;
[idx, clusterCenters] = kmeans(zData', numClusters);

% Initialize arrays for mean values
meanK = zeros(numClusters, 1);
meanSigma = zeros(numClusters, 1);
meanTorque = zeros(numClusters, 1);

% Compute mean k, sigma, and torque for each cluster
for i = 1:numClusters
    clusterIndices = (idx == i);
    meanK(i) = mean(xData(clusterIndices));
    meanSigma(i) = mean(yData(clusterIndices));
    meanTorque(i) = mean(zData(clusterIndices));
end

% Replot the 10 new data points
figure;
scatter3(meanK, meanSigma, meanTorque, 50, meanTorque, 'filled');
xlabel('k');
ylabel('sigma');
zlabel('torque');
title('Clustered 3D Scatter Plot');
colorbar
grid on;