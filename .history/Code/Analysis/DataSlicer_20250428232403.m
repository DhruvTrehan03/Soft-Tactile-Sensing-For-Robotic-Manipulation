clear;

% Define folder and diameters
folder ="C:\Users\dhruv\Soft-Tactile-Sensing-For-Robotic-Manipulation\Readings\40mm";
d = 10;

right_file = fullfile(folder, "10mm_right.mat");
truth_file = fullfile(folder, "10mm_truth.mat");
Left_Data = load(right_file).plotthis_right;
Load_Data = smoothdata(abs(load(truth_file).plotthis_truth));

% Process the data
EIT = Left_Data(:,2:end);
EIT = EIT(:,~all(EIT==0));
EIT_Time = Left_Data(:,1);
[pks, locs] = findpeaks(Load_Data(:,2), 'MinPeakHeight', 0.015);
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

closest_values = [40, 57, 72, 89, 109, 125, 139, 157];
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
figure()
plot(EIT, 'LineWidth', 2);
hom = mean(EIT(48, :));

mean_torque = mean(pks);
pause();
% Save testing data
for i = 1:length(closest_values)
    data = EIT(closest_values(i), :);
    data_diff = data - hom;
    save(sprintf("SavedVariables\\Diameter_Slices\\%dmm_Data_%i", d, i), "data_diff","mean_torque");
    disp(sprintf("Saved Data_%i for %dmm", i, d));
end



