folder = "..\Readings\2024-12-05_18-15"

load_file = fullfile(folder, "device1.mat");
Right_Data = load(load_file).Right_Data; % Device 3 data
Time = Right_Data(:,1);
EIT = Right_Data(:,2:end);
whos
%Convert Time from datenum (datenum(datetime('now', 'Format', 'HH:mm:ss.SSS'))) to seconds
Time = (Time - Time(1)) * 24 * 60 * 60; % Convert to seconds
D_Time = diff(Time);
mean_freq = mean(1./D_Time);
disp("Mean Frequency: " + num2str(mean_freq) + " Hz");

disp("Total Number of Readings: " + length(Time));
disp("Total Time: " + (Time(end) - Time(1)) + " seconds");
disp("Frequency: " + num2str(1/(Time(end) - Time(1)) * length(Time)) + " Hz");
