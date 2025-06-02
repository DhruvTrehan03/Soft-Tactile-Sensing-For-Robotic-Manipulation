folder = "..\Readings\2024-12-05_18-15"
load_file = fullfile(folder, "device1.mat");
Right_Data = load(load_file).Right_Data; % Device 3 data
Time = Right_Data(:,1);
EIT = Right_Data(:,2:end);

%Convert Time from datenum (datenum(datetime('now', 'Format', 'HH:mm:ss.SSS'))) to seconds
Time = (Time - min(Time)) * 24 * 60 * 60; % Convert to seconds

whos
plot(Time, EIT);