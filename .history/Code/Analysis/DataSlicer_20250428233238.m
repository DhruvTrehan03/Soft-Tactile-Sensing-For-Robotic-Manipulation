clear;

% Define folder and diameters
folder = "C:\Users\dhruv\Soft-Tactile-Sensing-For-Robotic-Manipulation\Readings\40mm";
d = 10;

right_file = fullfile(folder, "20mm_1_right.mat");
truth_file = fullfile(folder, "20mm_1_truth.mat");
Left_Data = load(right_file).plotthis_right;
Load_Data = smoothdata(abs(load(truth_file).plotthis_truth));

% Process the data
EIT = Left_Data(:, 2:end);
EIT = EIT(:, ~all(EIT == 0));
EIT_Time = Left_Data(:, 1);
[pks, locs] = findpeaks(Load_Data(:, 2), 'MinPeakHeight', 0.015);
torqueTimes = Load_Data(locs, 1);

% Estimate noise using the first 10 samples
noise_estimate = mean(EIT(1:10, :), 1);

% Remove noise from the data
EIT_denoised = EIT - noise_estimate;

% High-pass filter to remove DC component
filtered_data = EIT_denoised - mean(EIT_denoised, 1);

% Set a threshold to detect pulses
threshold = 0.02; % Adjust this value based on your data
pulse_indices = abs(filtered_data) > threshold;

% Highlight pulses in the data
highlighted_data = zeros(size(filtered_data));
highlighted_data(pulse_indices) = filtered_data(pulse_indices);

% Plot the original and highlighted pulses
figure();
plot(EIT_Time, filtered_data, 'b', 'LineWidth', 1.5); % Original filtered data
hold on;
plot(EIT_Time, highlighted_data, 'r', 'LineWidth', 1.5); % Highlighted pulses
title(sprintf("Pulse Detection for %dmm", d));
xlabel("Time");
ylabel("Amplitude");
legend("Filtered Data", "Highlighted Pulses");
hold off;



