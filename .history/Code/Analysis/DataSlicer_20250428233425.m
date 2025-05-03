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
highlighted_data = filtered_data .* pulse_indices;
highlighted_data(~pulse_indices) = NaN; % Set non-pulse data to NaN

% Ensure EIT_Time and torqueTimes are in the same range
if max(torqueTimes) > max(EIT_Time) || min(torqueTimes) < min(EIT_Time)
    warning("Torque times are out of range of EIT times. Check data alignment.");
end

% Find the closest indices in EIT_Time for torqueTimes
closest_values = arrayfun(@(t) find(EIT_Time >= t, 1, 'first'), torqueTimes);

% Handle cases where no match is found
closest_values(isnan(closest_values)) = length(EIT_Time);
closest_values = ceil(closest_values);
disp(sprintf("Closest values for %dmm: ", d));
disp(closest_values);

closest_values = [40, 57, 72, 89, 109, 125, 139, 157];
% Plot the truth data for verification
figure();
plot(EIT_Time, filtered_data, 'b', 'LineWidth', 1.5); % Original filtered data
hold on;
plot(EIT_Time, highlighted_data, 'r', 'LineWidth', 1.5); % Highlighted pulses
title(sprintf("Pulse Detection for %dmm", d));
xlabel("Time");
ylabel("Load");

% Plot the line to slice along for verification
figure();
hold on;
plot(mean(EIT_denoised, 2));
xline(closest_values, 'k--', 'LineWidth', 2);
title(sprintf("Sliced Data for %dmm (Denoised)", d));
xlabel("Sensor Index");
ylabel("Difference");
hold off;

% Plot the denoised EIT data
figure();
plot(EIT_denoised, 'LineWidth', 2);
hom = mean(EIT_denoised(1:20, :));

% Save testing data
mean_torque = mean(pks);
for i = 1:length(closest_values)
    data = EIT_denoised(closest_values(i), :);
    data_diff = data - hom;
    save(sprintf("SavedVariables\\Diameter_Slices\\%dmm_Data_%i", d, i), "data_diff", "mean_torque");
    disp(sprintf("Saved Data_%i for %dmm", i, d));
end



