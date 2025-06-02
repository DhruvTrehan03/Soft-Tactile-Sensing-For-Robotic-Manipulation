clear all;
close all;


%% Torque Analysis
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
% Display correlation between torque and sigma
disp('Correlation between Torque and Sigma:');
disp(corr(torque_sorted', sigma_sorted'));
% Display correlation between torque and sigma^2
disp('Correlation between Torque and Sigma^2:');
disp(corr(torque_sorted', sigma_sorted'.^2));
% Polyfit for sigma vs torque (order 1 and 2)
p1_sigma = polyfit(sigma_sorted, torque_sorted, 1);
% Calculate fitted values and normalized MSE for sigma vs torque
fitted_torque_p1_sigma = polyval(p1_sigma, sigma_sorted);
mse_p1_sigma = mean((torque_sorted - fitted_torque_p1_sigma).^2);
nmse_p1_sigma = mse_p1_sigma / var(torque_sorted);
% Display results for sigma vs torque
disp('Sigma vs Torque:');
disp(['Order 1 nMSE: ', num2str(nmse_p1_sigma)]);
%display correlation between torque and sigma
% Plot for sigma vs torque with polynomial fits (first instance)
figure;
scatter(sigma_sorted, torque_sorted, 50, torque_sorted,'DisplayName', 'Data');
% Create a custom colormap based on jet, excluding yellow/light orange
cmap = jet(256); % Get the jet colormap with 256 colors
cmap = cmap(1:180,200:256, :); % Keep only the first 200 colors, removing the yellow/light orange range
colormap(cmap); % Apply the custom colormap
colorbar; % Add a colorbar to indicate the mapping of color to torque_sorted values
hold on;
plot(sigma_sorted, fitted_torque_p1_sigma, '-r', 'DisplayName', ...
    ['Order 1 Fit: y = ', num2str(p1_sigma(1)), 'x + ', num2str(p1_sigma(2)), ...
    ', nMSE = ', num2str(nmse_p1_sigma)]);
xlabel('Sigma');
ylabel('Torque');
title('Sigma vs Torque with Polynomial Fits');
legend('show');
grid on;
% Ensure data is sorted for polyfit
[sigma_sorted, sort_idx] = sort(sigma_means);
torque_sorted = torque_means(sort_idx);
%Display correlation between torque and sigma
disp('Correlation between Torque and Sigma:');
disp(corr(torque_sorted, sigma_sorted));
%Display correlation between torque and sigma^2
disp('Correlation between Torque and Sigma^2:');
disp(corr(torque_sorted, sigma_sorted.^2));
% Polyfit for sigma vs torque (order 1 and 2)
p1_sigma = polyfit(sigma_sorted, torque_sorted, 1);
p2_sigma = polyfit(sigma_sorted, torque_sorted, 2);
% Calculate zitted values and normalized MSE for sigma vs torque
fitted_torque_p1_sigma = polyval(p1_sigma, sigma_sorted);
fitted_torque_p2_sigma = polyval(p2_sigma, sigma_sorted);
mse_p1_sigma = mean((torque_sorted - fitted_torque_p1_sigma).^2);
mse_p2_sigma = mean((torque_sorted - fitted_torque_p2_sigma).^2);
nmse_p1_sigma = mse_p1_sigma / var(torque_sorted);
nmse_p2_sigma = mse_p2_sigma / var(torque_sorted);
% Display results for sigma vs torque
disp('Sigma vs Torque:');
disp(['Order 1 nMSE: ', num2str(nmse_p1_sigma)]);
disp(['Order 2 nMSE: ', num2str(nmse_p2_sigma)]);
% Plot for sigma vs torque with polynomial fits (second instance)
figure;
scatter(sigma_sorted, torque_sorted, 50, torque_sorted, 'filled', 'DisplayName', 'Data');
colormap(jet); % High contrast colormap suitable for white background
colorbar; % Add a colorbar to indicate the mapping of color to torque_sorted values
hold on;
plot(sigma_sorted, fitted_torque_p1_sigma, '-r', 'DisplayName', ...
    ['Order 1 Fit: y = ', num2str(p1_sigma(1)), 'x + ', num2str(p1_sigma(2)), ...
    ', nMSE = ', num2str(nmse_p1_sigma)]);
plot(sigma_sorted, fitted_torque_p2_sigma, '--g', 'DisplayName', ...
    ['Order 2 Fit: y = ', num2str(p2_sigma(1)), 'x^2 + ', num2str(p2_sigma(2)), ...
    'x + ', num2str(p2_sigma(3)), ', nMSE = ', num2str(nmse_p2_sigma)]);
xlabel('Sigma');
ylabel('Torque');
title('Sigma vs Torque with Polynomial Fits');
legend('show');
grid on;

%% Diameter Analysis
% Saved as diameter_value, k_values(max_k_idx), sigma_values(max_sigma_idx), max_val
load('maxima_data_Diam.mat');

figure;
scatter3(maxima_matrix(:, 2), maxima_matrix(:, 3), maxima_matrix(:, 1), 50, maxima_matrix(:, 1), 'filled'); 
xlabel('k');
ylabel('Sigma');
zlabel('Diameter');
title('Maxima of k vs Sigma with Diameter as Color Gradient');
colorbar;
grid on;

% K vs d
% Ensure data is sorted for polyfit
[k_sorted, sort_idx] = sort(maxima_matrix(:, 2));
d_sorted = maxima_matrix(sort_idx, 1);
% Display correlation between d and k
disp('Correlation between Diameter and k:');
disp(corr(d_sorted, k_sorted));

%plot for k vs diam with polynomial fits
figure;
scatter(k_sorted, d_sorted, 50, d_sorted, 'filled', 'DisplayName', 'Data');
colormap(jet); % High contrast colormap suitable for white background
colorbar; % Add a colorbar to indicate the mapping of color to d_sorted values

hold on;
p1_k = polyfit(k_sorted, d_sorted, 1);
fitted_d_p1_k = polyval(p1_k, k_sorted);
mse_p1_k = mean((d_sorted - fitted_d_p1_k).^2);
nmse_p1_k = mse_p1_k / var(d_sorted);
disp('Diameter vs k:');
disp(['Order 1 nMSE: ', num2str(nmse_p1_k)]);
plot(k_sorted, fitted_d_p1_k, '-r', 'DisplayName', ...
    ['Order 1 Fit: y = ', num2str(p1_k(1)), 'x + ', num2str(p1_k(2)), ...
    ', nMSE = ', num2str(nmse_p1_k)]);
xlabel('k');
ylabel('Diameter');
title('k vs Diameter with Polynomial Fits');
legend('show');






