clear all;
close all;

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

% Ensure data is sorted for polyfit
[sigma_sorted, sort_idx] = sort(maxima_sigma);
torque_sorted = torque_values(sort_idx);

% Polyfit for sigma vs torque (order 1 and 2)
p1_sigma = polyfit(sigma_sorted, torque_sorted, 1);

% Calculate fitted values and MSE for sigma vs torque
fitted_torque_p1_sigma = polyval(p1_sigma, sigma_sorted);
mse_p1_sigma = mean((torque_sorted - fitted_torque_p1_sigma).^2);

% Display results for sigma vs torque
disp('Sigma vs Torque:');
disp(['Order 1 MSE: ', num2str(mse_p1_sigma)]);


% Plot for sigma vs torque with polynomial fits
figure;
plot(sigma_sorted, torque_sorted, 'o', 'DisplayName', 'Data');
hold on;
plot(sigma_sorted, fitted_torque_p1_sigma, '-r', 'DisplayName', 'Order 1 Fit');
xlabel('Sigma');
ylabel('Torque');
title('Sigma vs Torque with Polynomial Fits');
legend('show');
grid on;

% Ensure data is sorted for polyfit
[sigma_sorted, sort_idx] = sort(sigma_means);
torque_sorted = torque_means(sort_idx);

% Polyfit for sigma vs torque (order 1 and 2)
p1_sigma = polyfit(sigma_sorted, torque_sorted, 1);
p2_sigma = polyfit(sigma_sorted, torque_sorted, 2);

% Calculate fitted values and MSE for sigma vs torque
fitted_torque_p1_sigma = polyval(p1_sigma, sigma_sorted);
fitted_torque_p2_sigma = polyval(p2_sigma, sigma_sorted);
mse_p1_sigma = mean((torque_sorted - fitted_torque_p1_sigma).^2);
mse_p2_sigma = mean((torque_sorted - fitted_torque_p2_sigma).^2);

% Display results for sigma vs torque
disp('Sigma vs Torque:');
disp(['Order 1 MSE: ', num2str(mse_p1_sigma)]);
disp(['Order 2 MSE: ', num2str(mse_p2_sigma)]);

% Plot for sigma vs torque with polynomial fits
figure;
plot(sigma_sorted, torque_sorted, 'o', 'DisplayName', 'Data');
hold on;
plot(sigma_sorted, fitted_torque_p1_sigma, '-r', 'DisplayName', 'Order 1 Fit');
plot(sigma_sorted, fitted_torque_p2_sigma, '--g', 'DisplayName', 'Order 2 Fit');
xlabel('Sigma');
ylabel('Torque');
title('Sigma vs Torque with Polynomial Fits');
legend('show');
grid on;

% Display equations and MSE on the plot
text(min(sigma_sorted), max(torque_sorted), ...
    {['Order 1: y = ' num2str(p1_sigma(1)) 'x + ' num2str(p1_sigma(2))], ...
     ['MSE: ' num2str(mse_p1_sigma)]}, ...
    'Color', 'red', 'VerticalAlignment', 'top');

text(min(sigma_sorted), max(torque_sorted) - 0.1 * range(torque_sorted), ...
    {['Order 2: y = ' num2str(p2_sigma(1)) 'x^2 + ' num2str(p2_sigma(2)) 'x + ' num2str(p2_sigma(3))], ...
     ['MSE: ' num2str(mse_p2_sigma)]}, ...
    'Color', 'green', 'VerticalAlignment', 'top');











