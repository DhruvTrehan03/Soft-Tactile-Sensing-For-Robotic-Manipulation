% Example matrix (rows x columns)
data = load('C:\Users\dhruv\Soft-Tactile-Sensing-For-Robotic-Manipulation\Readings\5ShapeTest.mat').plotthis_right(2:end,:);

% Compute the range of each column
col_ranges = max(data) - min(data);

% Define a threshold for removal (e.g., remove columns where range < 20)
threshold = 0.2;
columns_to_keep = col_ranges >= threshold;

% Remove columns where range is below the threshold
filtered_data = data(:, columns_to_keep);

% Display results
disp('Original Matrix:');
disp(data);
disp('Filtered Matrix:');
disp(filtered_data);
figure()
plot(filtered_data)
figure()
plot(filtered_data(53,:)-filtered_data(2,:))