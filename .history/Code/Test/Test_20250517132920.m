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

means = zeros(4,1);
stds = zeros(4,1);


load("..\Readings\40mm\10mm_1_truth.mat");
plot(plotthis_truth(:,1), plotthis_truth(:,2:end));

%find peaks list
[pks, locs] = findpeaks(plotthis_truth(:,2), 'MinPeakHeight', 0.005);

%find mean of peaks and std
means(1) = mean(pks);
stds(1) = std(pks);

load("..\Readings\40mm\20mm_1_truth.mat");
plot(plotthis_truth(:,1), plotthis_truth(:,2:end));

%find peaks list
[pks, locs] = findpeaks(plotthis_truth(:,2), 'MinPeakHeight', 0.005);
%find mean of peaks and std
means(2) = mean(pks);
stds(2) = std(pks);

load("..\Readings\40mm\30mm_1_truth.mat");
plot(plotthis_truth(:,1), plotthis_truth(:,2:end));

%find peaks list
[pks, locs] = findpeaks(plotthis_truth(:,2), 'MinPeakHeight', 0.005);
%find mean of peaks and std
means(3) = mean(pks);
stds(3) = std(pks);

load("..\Readings\40mm\40mm_1_truth.mat");
plot(plotthis_truth(:,1), plotthis_truth(:,2:end));

%find peaks list
[pks, locs] = findpeaks(plotthis_truth(:,2), 'MinPeakHeight', 0.005);
%find mean of peaks and std
means(4) = mean(pks);
stds(4) = std(pks);

disp('Means:');
for i = 1:length(means)
    fprintf('%.4f \\pm %.4f,\n', means(i), stds(i));
end

% Create a bar plot of means with error bars of stds
figure;
bar(means);
xlabel('Diameter');
xticklabels = {'10mm', '20mm', '30mm', '40mm'};
ylabel('Torque Value');
title('Torque Values with Error Bars');
hold on;
errorbar(means, stds, 'k', 'LineStyle', 'none', 'LineWidth', 1);
hold off;
% Set x-tick labels