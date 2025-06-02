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
diameters = [10, 20, 30, 40]; % Example diameters

%load data for each diameter
means = zeros(length(diameters), 1);
for i = 1:length(diameters)
    % Load data for the current diameter
    file_path = fullfile("SavedVariables\Diameter_Slices\", sprintf('%dmm*.mat', diameters(i)));
    files = dir(file_path);
    % Initialize variables to store torque values and data
    mean_torque = zeros(1, length(files));
    % Loop through each file
    for j = 1:length(files)
        % Load the data
        file_name = fullfile(files(j).folder, files(j).name);
        data = load(file_name);
        % Extract torque value and data
        mean_torque(j) = data.mean_torque;
    end
    means(i) = mean(mean_torque);

    % Display the mean and standard deviation
    fprintf('Diameter: %d mm, Mean Torque: %.4f \n', diameters(i), means(i));
end

load("..\Readings\40mm\10mm_2_truth.mat");
plot(plotthis_truth(:,1), plotthis_truth(:,2:end));
