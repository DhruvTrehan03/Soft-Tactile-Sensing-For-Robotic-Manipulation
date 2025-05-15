clear;

% Define folder and diameters
folder = "..\Readings\2024-12-05_18-15";

% Load data from device1 and device3
left_file = fullfile(folder, "device1.mat");
load_file = fullfile(folder, "device3.mat");

Left_Data = load(left_file); % Device 1 data
Load_Data = load(load_file); % Device 3 data

% Process the data
EIT = Left_Data(:,~all(Left_Data==0));
EIT_O = EIT;
EIT = smoothdata(EIT(26:171,2:end), 'gaussian', 9);

EIT_Time = Left_Data(:,1);
[pks, locs] = findpeaks(Load_Data(:,2), 'MinPeakHeight', 0.005);
disp(pks);
torqueTimes = Load_Data(locs,1);

% Ensure EIT_Time and torqueTimes are in the same range
if max(torqueTimes) > max(EIT_Time) || min(torqueTimes) < min(EIT_Time)
    warning("Torque times are out of range of EIT times. Check data alignment.");
end

% Find the closest indices in EIT_Time for torqueTimes
closest_values = arrayfun(@(t) find(EIT_Time >= t, 1, 'first'), torqueTimes);

% Handle cases where no match is found
closest_values(isnan(closest_values)) = length(EIT_Time);
closest_values = ceil(closest_values);
disp("Closest values: ");
disp(closest_values);

% Plot the truth data for verification
figure();
plot(Load_Data(:,1), Load_Data(:,2));
hold on;
xline(torqueTimes, 'k--', 'LineWidth', 2);
title("Truth Data");
xlabel("Time");
ylabel("Load");

% Plot the line to slice along for verification
figure();
hold on;
plot(mean(EIT,2));
xline(closest_values, 'k--', 'LineWidth', 2);
title("Sliced Data");
xlabel("Sensor Index");
ylabel("Difference");
hold off;

hom = mean(EIT_O(1:5, 2:end),1); % Homogeneous data for the first two rows

% Plot the EIT data for each slice
figure();
hold on;
for i = 1:length(closest_values)
    plot(calculate_envelope(abs(EIT(closest_values(i), :)-hom),30) * max(abs(EIT(closest_values(i), :)-hom)), 'DisplayName', sprintf("Slice %d", i));
end
title("EIT Data for Slices");
mean_torque = mean(pks);
pause();

% % Save testing data
% for i = 1:length(closest_values)
%     data = EIT(closest_values(i), :);
%     data_diff = abs(data - hom);
%     torque_value = pks(i); % Corresponding torque value
%     save(sprintf("SavedVariables\\Slices\\Data_Slice_%i", i), "data_diff", "torque_value");
%     disp(sprintf("Saved Data_Slice_%i with torque %.4f", i, torque_value));
% end

function env = calculate_envelope(data, smooth_coeff)
    % Extend the signal to reduce edge effects
    data_ext = [data; data; data];  % Triplicate the data

    % Compute the envelope of the extended signal
    [env_ext, ~] = envelope(data_ext, 10, 'peak');

    % Extract only the middle section to avoid edge artifacts
    N = length(data);
    env = env_ext(N+1:2*N);

    % Smooth the envelope
    windowSize = smooth_coeff; % Adjust this value as needed
    b = (1/windowSize) * ones(1, windowSize);
    a = 1;
    env = filter(b, a, env);

    % Normalize the envelope
    env = (env - min(env)) / (max(env) - min(env));
end
