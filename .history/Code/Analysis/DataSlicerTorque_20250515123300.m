clear;

% Define folder and diameters
folder = "..\Readings\2024-12-05_18-15";

% Load data from device1 and device3
left_file = fullfile(folder, "device1.mat");
load_file = fullfile(folder, "device3.mat");
Left_Data = load(left_file).Right_Data; % Device 1 data
Load_Data = load(load_file).Load_Data; % Device 3 data

% Process the data
EIT = Left_Data(:,2:end);
EIT_Time = Left_Data(:,1);

EIT = EIT(:,~all(Left_Data==0));
EIT_O = EIT;
% EIT = smoothdata(EIT(26:171,2:end), 'gaussian', 9);

[pks, locs] = findpeaks(Load_Data(:,2), 'MinPeakHeight', 0.005);
disp(pks);
torqueTimes = Load_Data(locs,1);

% Ensure EIT_Time and torqueTimes are in the same range
if max(torqueTimes) > max(EIT_Time) || min(torqueTimes) < min(EIT_Time)
    warning("Torque times are out of range of EIT times. Check data alignment.");
end

% Plot the truth data for verifiation
figure();
plot(Load_Data(:,1), Load_Data(:,2));
hold on;
xline(torqueTimes, 'k--', 'LineWidth', 0.1);
title("Truth Data");
xlabel("Time");
ylabel("Load");

% Plot the EIT data with peaks
figure();
plot(EIT_Time, EIT(:,1)); % Plot the first column of EIT as an example
hold on;

% Find the closest EIT times to the torque times
closest_indices = arrayfun(@(t) find(abs(EIT_Time - t) == min(abs(EIT_Time - t)), 1), torqueTimes);
closest_EIT_times = EIT_Time(closest_indices);

% Mark the peaks on the EIT plot
scatter(closest_EIT_times, EIT(closest_indices, 1), 'r', 'filled'); % Mark peaks on the first column of EIT
title("EIT Data with Peaks");
xlabel("Time");
ylabel("EIT Signal");
legend("EIT Data", "Peaks");

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
