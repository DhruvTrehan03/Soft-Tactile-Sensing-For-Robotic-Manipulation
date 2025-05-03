clear;

% Define folder and diameters
folder ="C:\Users\dhruv\Soft-Tactile-Sensing-For-Robotic-Manipulation\Readings\40mm";

d = 40;

right_file = fullfile(folder, sprintf("%dmm_1_right.mat",d));
truth_file = fullfile(folder, sprintf("%dmm_1_truth.mat",d));
Left_Data = load(right_file).plotthis_right;
Load_Data = smoothdata(abs(load(truth_file).plotthis_truth));

% Process the data
EIT = Left_Data(:,~all(Left_Data==0));
EIT_O = EIT;
EIT = smoothdata(EIT(25:173,2:end), 'gaussian', 7);

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
disp(sprintf("Closest values for %dmm: ", d));
disp(closest_values);

closest_values = [15, 43, 66, 108, 127, 139]; % Adjusted for the EIT data range
% Plot the truth data for verification
figure();
plot(Load_Data(:,1), Load_Data(:,2));
hold on;
xline(torqueTimes, 'k--', 'LineWidth', 2);
title(sprintf("Truth Data for %dmm", d));
xlabel("Time");
ylabel("Load");
% Plot the line to slice along for verification
figure();
hold on;
plot(mean(EIT,2));
xline(closest_values, 'k--', 'LineWidth', 2);
title(sprintf("Sliced Data for %dmm", d));
xlabel("Sensor Index");
ylabel("Difference");
hold off;


hom = mean(EIT_O(1:5, 2:end),1); % Homogeneous data for the first two rows

%Plot the EIT data for each slice
figure();
hold on;
for i = 1:length(closest_values)
    plot(abs(EIT(closest_values(i), :)-hom), 'DisplayName', sprintf("Slice %d", i));
end
title(sprintf("EIT Data for %dmm", d));
mean_torque = mean(pks);
pause();
% Save testing data
for i = 1:length(closest_values)
    data = EIT(closest_values(i), :);
    data_diff = abs(data - hom);
    save(sprintf("SavedVariables\\Diameter_Slices\\%dmm_Data_%i", d, i), "data_diff","mean_torque");
    disp(sprintf("Saved Data_%i for %dmm", i, d));
end



