load('maxima_data.mat');
scatter3(maxima_k, maxima_sigma, torque_values, 20, maxima_corr, 'filled');
xlabel('k');
ylabel('Sigma');
zlabel('Torque');
title('Maxima of k vs Sigma with Torque as Color Gradient');
colorbar;
grid on;

% K means clustering of torque into 10 clusters, taking mean of each cluster k, sigma and corr too
[cluster_idx, cluster_centers] = kmeans(torque_values', 10);
% Take the mean of each cluster
torque_means = zeros(10, 1);
sigma_means = zeros(10, 1);
k_means = zeros(10, 1);
corr_means = zeros(10, 1);
for i = 1:10
    torque_means(i) = mean(torque_values(cluster_idx == i));
    sigma_means(i) = mean(maxima_sigma(cluster_idx == i));
    k_means(i) = mean(maxima_k(cluster_idx == i));
    corr_means(i) = mean(maxima_corr(cluster_idx == i));
end

% Plot the clusters
figure;
scatter3(k_means, sigma_means, torque_means, 50, corr_means, 'filled');
xlabel('k');
ylabel('Sigma');
zlabel('Torque');
title('Clustered Maxima of k vs Sigma with Torque as Color Gradient');
colorbar;
grid on;

figure;
scatter3(k_means, sigma_means.^2, torque_means, 50, corr_means, 'filled');
xlabel('k');
ylabel('Sigma^2');
zlabel('Torque');
title('Clustered Maxima of k vs Sigma with Torque as Color Gradient');
colorbar;
grid on;

figure;
scatter(sigma_means, torque_means, 50, corr_means, 'filled');
%line of best fit
p = polyfit(sigma_means, torque_means, 2);
y = polyval(p, sort(sigma_means));
display(p);
hold on;
plot(sort(sigma_means), y, 'r-');
hold off;
xlabel('Sigma');
ylabel('Torque');
title('Clustered Maxima of k vs Sigma with Torque as Color Gradient, MSE = ' + num2str(mean((y - torque_means).^2)));
legend('Torque', 'Line of Best Fit equation: y = ' + num2str(p(1)) + 'x^2 + ' + num2str(p(2)) + 'x + ' + num2str(p(3)));
colorbar;
grid on;

% Repeat for sigma^2
figure;
scatter(sigma_means.^2, torque_means, 50, corr_means, 'filled');
%line of best fit
p = polyfit(sigma_means.^2, torque_means, 2);
y = polyval(p, sigma_means.^2);
hold on;
plot(sigma_means.^2, y, 'r-');
hold off;
xlabel('Sigma^2');
ylabel('Torque');
title('Clustered Maxima of k vs Sigma with Torque as Color Gradient, MSE = ' + num2str(mean((y - torque_means).^2)));
legend('Torque', 'Line of Best Fit equation: y = ' + num2str(p(1)) + 'x^2 + ' + num2str(p(2)) + 'x + ' + num2str(p(3)));
colorbar;
grid on;

