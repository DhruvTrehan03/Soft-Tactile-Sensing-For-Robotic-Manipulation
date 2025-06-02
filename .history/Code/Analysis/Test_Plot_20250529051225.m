clear all;
close all;


%% Torque Analysis
load('maxima_data.mat');
% take every 11th point and move them all to the end of the vector

move_idx = 1:11:length(torque_values);

%add the indices to the end of the vector
torque_values= [torque_values, torque_values(move_idx)];
torque_values(move_idx) = []; % Remove every 11th point

maxima_k = [maxima_k, maxima_k(move_idx)];
maxima_k(move_idx) = []; % Remove every 11th point
maxima_sigma = [maxima_sigma, maxima_sigma(move_idx)];
maxima_sigma(move_idx) = []; % Remove every 11th point
maxima_corr = [maxima_corr, maxima_corr(move_idx)];
maxima_corr(move_idx) = []; % Remove every 11th point
% maxima_sigma = maxima_sigma.^2; % Square the sigma values for analysis
scatter(maxima_sigma,torque_values, 75, torque_values, 'filled');
xlabel('k');
ylabel('Sigma');
title('Maxima of k vs Sigma with Torque as Color Gradient');
%Lable the colorbar as Torque
cbar = colorbar;
cbar.Label.String = 'Torque';
cbar.Label.FontSize = 14;
cbar.FontSize = 14;
colormap(jet); % Use a colormap that is suitable for continuous data
caxis([min(torque_values) max(torque_values)]);

grid on;
% K means clustering of torque into 10 clusters, taking mean of each cluster k, sigma and corr too
% [cluster_idx, cluster_centers] = kmeans(torque_values', 10);
%instead of kmeans do every 10th point
% Create a vector of cluster indices


num_clusters = 10;
cluster_idx = zeros(size(torque_values));
for i = 1:num_clusters
    cluster_idx(i:num_clusters:end) = i; % Assign every 10th point to a cluster
end

% Take the mean of each cluster
torque_means = zeros(10, 1);
torque_stds = zeros(10, 1);
sigma_means = zeros(10, 1);
sigma_stds = zeros(10, 1);
k_means = zeros(10, 1);
corr_means = zeros(10, 1);
for i = 1:10
    torque_means(i) = mean(torque_values(cluster_idx == i));
    torque_stds(i) = std(torque_values(cluster_idx == i));
    sigma_means(i) = mean(maxima_sigma(cluster_idx == i));
    k_means(i) = mean(maxima_k(cluster_idx == i));
    corr_means(i) = mean(maxima_corr(cluster_idx == i));
    sigma_stds(i) = std(maxima_sigma(cluster_idx == i));
end
figure;
% Plot the means of torque 
plot(torque_values);

% Plot the clusters
figure;
scatter(sigma_means, torque_means, 75, 'filled');
xlabel('Sigma');
ylabel('Torque');
zlabel('Torque');
title('Clustered Maxima of k vs Sigma with Torque as Color Gradient');
grid on;
disp(torque_means);
disp('Correlation between Torque and Sigma:');
% disp(corr(sigma_means',torque_means'));
% % Ensure data is sorted for polyfit
% [sigma_sorted, sort_idx] = sort(maxima_sigma);
% torque_sorted = torque_values(sort_idx);
% % Display correlation between torque and sigma
% disp('Correlation between Torque and Sigma:');
% disp(corr(torque_sorted', sigma_sorted'));
% % Display correlation between torque and sigma^2
% disp('Correlation between Torque and Sigma^2:');
% disp(corr(torque_sorted', sigma_sorted'.^2));
% % Polyfit for sigma vs torque (order 1 and 2)
% p1_sigma = polyfit(sigma_sorted, torque_sorted, 1);
% % Calculate fitted values and normalized MSE for sigma vs torque
% fitted_torque_p1_sigma = polyval(p1_sigma, sigma_sorted);
% mse_p1_sigma = mean((torque_sorted - fitted_torque_p1_sigma).^2);
% nmse_p1_sigma = mse_p1_sigma / var(torque_sorted);
% % Display results for sigma vs torque
% disp('Sigma vs Torque:');
% disp(['Order 1 nMSE: ', num2str(nmse_p1_sigma)]);
% %display correlation between torque and sigma
% % Plot for sigma vs torque with polynomial fits (first instance)
% figure;
% scatter(sigma_sorted, torque_sorted, 50, 'filled','DisplayName', 'Data');
% hold on;
% plot(sigma_sorted, fitted_torque_p1_sigma, '-r', 'DisplayName', ...
%     ['Order 1 Fit: y = ', num2str(p1_sigma(1)), 'x + ', num2str(p1_sigma(2)), ...
%     ', nMSE = ', num2str(nmse_p1_sigma)]);
% xlabel('Sigma');
% ylabel('Torque');
% title('Sigma vs Torque with Polynomial Fits');
% grid on;

% %Plot means of sigma as bar plot with variance as error bars
% figure;
% [sorted_sigma_means,idx] = sort(sigma_means);
% sorted_sigma_stds = sigma_stds(idx);
% bar(sorted_sigma_means.^2, 'FaceColor', [0.2 0.6 0.8], 'EdgeColor', 'k', 'LineWidth', 1.5);
% % Set the x-ticks to the cluster indices
% hold on;
% errorbar(sorted_sigma_means.^2,sorted_sigma_stds, 'k.', 'LineWidth', 1.5);
% ylim([1.95 2.85]);

% xlabel('Torque Number');
% ylabel('Mean Sigma^2');

% %Same Bar plot with torque means
% figure;
% [sorted_torque_means,idx_torque] = sort(torque_means);
% bar(sorted_torque_means, 'FaceColor', [0.2 0.6 0.8], 'EdgeColor', 'k', 'LineWidth', 1.5);
% hold on;
% errorbar(sorted_torque_means, torque_stds(idx_torque), 'k.', 'LineWidth', 1.5);
% xlabel('Torque Number');
% ylabel('Mean Torque');
% title('Mean Torque for Each Cluster with Error Bars');



% % Ensure data is sorted for polyfit
% [sigma_sorted, sort_idx] = sort(sigma_means);
% torque_sorted = torque_means(sort_idx);
% %Display correlation between torque and sigma
% disp('Correlation between Torque and Sigma:');
% disp(corr(torque_sorted, sigma_sorted));
% %Display correlation between torque and sigma^2
% disp('Correlation between Torque and Sigma^2:');
% disp(corr(torque_sorted, sigma_sorted.^2));
% % Polyfit for sigma vs torque (order 1 and 2)
% p1_sigma = polyfit(sigma_sorted, torque_sorted, 1);
% p2_sigma = polyfit(sigma_sorted, torque_sorted, 2);
% % Calculate zitted values and normalized MSE for sigma vs torque
% fitted_torque_p1_sigma = polyval(p1_sigma, sigma_sorted);
% fitted_torque_p2_sigma = polyval(p2_sigma, sigma_sorted);
% mse_p1_sigma = mean((torque_sorted - fitted_torque_p1_sigma).^2);
% mse_p2_sigma = mean((torque_sorted - fitted_torque_p2_sigma).^2);
% nmse_p1_sigma = mse_p1_sigma / var(torque_sorted);
% nmse_p2_sigma = mse_p2_sigma / var(torque_sorted);
% % Display results for sigma vs torque
% disp('Sigma vs Torque:');
% disp(['Order 1 nMSE: ', num2str(nmse_p1_sigma)]);
% disp(['Order 2 nMSE: ', num2str(nmse_p2_sigma)]);
% % Plot for sigma vs torque with polynomial fits (second instance)
% figure;
% scatter(sigma_sorted, torque_sorted, 50, 'filled', 'DisplayName', 'Data');
% hold on;
% % plot(sigma_sorted, fitted_torque_p1_sigma, '-r', 'DisplayName', ...
% %     ['Order 1 Fit: y = ', num2str(p1_sigma(1)), 'x + ', num2str(p1_sigma(2)), ...
% %     ', nMSE = ', num2str(nmse_p1_sigma)]);
% % plot(sigma_sorted, fitted_torque_p2_sigma, '--g', 'DisplayName', ...
% %     ['Order 2 Fit: y = ', num2str(p2_sigma(1)), 'x^2 + ', num2str(p2_sigma(2)), ...
% %     'x + ', num2str(p2_sigma(3)), ', nMSE = ', num2str(nmse_p2_sigma)]);
% xlabel('Sigma');
% ylabel('Torque');
% title('Sigma vs Torque with Polynomial Fits');
% grid on;



% %% Diameter Analysis
% % Saved as diameter_value, k_values(max_k_idx), sigma_values(max_sigma_idx), max_val
% load('maxima_data_Diam.mat');
% figure;
% load('maxima_data.mat');
% scatter(maxima_matrix(:, 2), maxima_matrix(:, 3), 75, maxima_matrix(:, 1), 'filled');
% xlabel('k');
% ylabel('Sigma');
% title('Characteristic K and Sigma for all Diameters');
% %Lable the colorbar as Torque
% cbar = colorbar;
% cbar.Label.String = 'Diameter';
% cbar.Label.FontSize = 14;
% cbar.FontSize = 14;
% colormap(jet); % Use a colormap that is suitable for continuous data
% caxis([min(maxima_matrix(:,1)) max(maxima_matrix(:, 1))]);
% grid on;


% % K vs d
% % Ensure data is sorted for polyfit
% [k_sorted, sort_idx] = sort(maxima_matrix(:, 2));
% d_sorted = maxima_matrix(sort_idx, 1);
% % Display correlation between d and k
% disp('Correlation between Diameter and k:');
% disp(corr(d_sorted, k_sorted));

% %plot for k vs diam with polynomial fits
% figure;
% scatter(k_sorted, d_sorted, 50, 'filled', 'DisplayName', 'Data');
% grid on;
% hold on;
% p1_k = polyfit(k_sorted, d_sorted, 1);
% fitted_d_p1_k = polyval(p1_k, k_sorted);
% mse_p1_k = mean((d_sorted - fitted_d_p1_k).^2);
% nmse_p1_k = mse_p1_k / var(d_sorted);
% disp('Diameter vs k:');
% disp(['Order 1 nMSE: ', num2str(nmse_p1_k)]);
% % plot(k_sorted, fitted_d_p1_k, '-r', 'DisplayName', ...
% %     ['Order 1 Fit: y = ', num2str(p1_k(1)), 'x + ', num2str(p1_k(2)), ...
% %     ', nMSE = ', num2str(nmse_p1_k)]);
% xlabel('k');
% ylabel('Diameter');
% title('k vs Diameter with Polynomial Fits');


% % Compute mean and std of sigma for each unique diameter
% unique_diameters = unique(maxima_matrix(:, 1));
% sigma_means_diameter = zeros(size(unique_diameters));
% sigma_stds_diameter = zeros(size(unique_diameters));

% for i = 1:length(unique_diameters)
%     sigma_values_for_diameter = maxima_matrix(maxima_matrix(:, 1) == unique_diameters(i), 3);
%     sigma_means_diameter(i) = mean(sigma_values_for_diameter);
%     sigma_stds_diameter(i) = std(sigma_values_for_diameter);
%     torque_values_for_diameter = maxima_matrix(maxima_matrix(:, 1) == unique_diameters(i), 4);
%     mean_torque_for_diameter(i) = mean(torque_values_for_diameter);
%     std_torque_for_diameter(i) = std(torque_values_for_diameter);
%     k_values_for_diameter = maxima_matrix(maxima_matrix(:, 1) == unique_diameters(i), 2);
%     mean_k_for_diameter(i) = mean(k_values_for_diameter);
%     std_k_for_diameter(i) = std(k_values_for_diameter);
% end

% %Scatter of mean k vs mean diameter
% figure;
% scatter(mean_k_for_diameter, unique_diameters, 75, 'filled', 'DisplayName', 'Data');
% xlabel('Mean k');
% ylabel('Mean Diameter');
% title('Mean k vs Mean Diameter');
% grid on;
% hold on;


% % Bar plot of mean sigma with error bars for each diameter
% figure;
% bar(unique_diameters, sigma_means_diameter, 'FaceColor', [0.2 0.6 0.8], 'EdgeColor', 'k', 'LineWidth', 1.5);
% hold on;
% errorbar(unique_diameters, sigma_means_diameter, sigma_stds_diameter, 'k.', 'LineWidth', 1.5);
% xlabel('Diameter');
% ylabel('Mean Sigma');
% title('Mean Sigma for Each Diameter with Error Bars');
% grid on;

% figure;
% % Bar plot of mean torque with error bars for each diameter
% bar(unique_diameters, mean_torque_for_diameter, 'FaceColor', [0.2 0.6 0.8], 'EdgeColor', 'k', 'LineWidth', 1.5);
% hold on;
% errorbar(unique_diameters, mean_torque_for_diameter, std_torque_for_diameter, 'k.', 'LineWidth', 1.5);
% xlabel('Diameter'); 
% ylabel('Mean Torque');
% title('Mean Torque for Each Diameter with Error Bars');
% grid on;