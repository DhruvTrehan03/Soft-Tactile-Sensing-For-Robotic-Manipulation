folder = "..\Readings\2024-12-05_18-15"
load_file = fullfile(folder, "device1.mat");
Right_Data = load(load_file).Right_Data; % Device 3 data
Time = Right_Data(:,1);
EIT = Right_Data(:,2:end);

%Convert Time from datetime to seconds
Time = seconds(Time - Time(1));

whos
plot(Time, EIT);