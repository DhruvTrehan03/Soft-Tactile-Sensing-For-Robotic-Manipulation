% load("..\Readings\2024-12-05_18-15" + "\device1.mat");
% load("..\Readings\2024-12-05_18-15" + "\device3.mat");

% plot(Load_Data(:,1), Load_Data(:,2:end));

% %find peaks list 
% [pks, locs] = findpeaks(Load_Data(:,2), 'MinPeakHeight', 0.005);

% %kmeans to find 10 clusters#
% [idx, C] = kmeans(pks, 10);
% % Create a figure
% figure;
% % Create a scatter plot of the data points
% scatter(1:length(pks), pks, 10, idx, 'filled');
% % Add a colorbar
% colorbar;


% %Find variance of each cluster and plot with error bars
% % Calculate the variance of each cluster
% cluster_variances = zeros(1, 10);
% for i = 1:10
%     cluster_variances(i) = std(pks(idx == i));
% end


% %Sort means and std
% [~, sorted_indices] = sort(C);
% std_sorted = cluster_variances(sorted_indices);
% C_sorted = C(sorted_indices);

% for i = 1:length(C_sorted)
%     fprintf('%.4f \\pm %.4f\n', C_sorted(i), std_sorted(i));
% end


% % Create a bar plot of C with error  bars of cluster_variances
% figure;
% bar(C_sorted);
% xlabel('Torque Number');
% ylabel('Torque Value');
% title('Torque Values with Error Bars');
% hold on;
% errorbar(C_sorted, std_sorted, 'k', 'LineStyle', 'none', 'LineWidth', 1);
% hold off;


clear;

load("SavedVariables\Diameter_Slices\10mm_Data_1.mat");

whos;