load('maxima_data.mat');
scatter3(maxima_k, maxima_sigma, torque_values, 20, maxima_corr, 'filled');
xlabel('k');
ylabel('Sigma');
zlabel('Torque');
title('Maxima of k vs Sigma with Torque as Color Gradient');
colorbar;
grid on;

% K means clustering of torque into 10 clusters, taking mean of each cluster k, sigma and corr too
[cluster_idx, cluster_centers] = kmeans(torque_values, 10);
% Plot the clusters
figure;
hold on;
for i = 1:10
    cluster_points = find(cluster_idx == i);
    scatter3(maxima_k(cluster_points), maxima_sigma(cluster_points), torque_values(cluster_points), 20, maxima_corr(cluster_points), 'filled');
end