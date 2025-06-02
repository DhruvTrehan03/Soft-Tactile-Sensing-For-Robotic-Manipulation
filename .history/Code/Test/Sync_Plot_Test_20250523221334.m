folder = "..\Readings\2024-12-05_18-15"
load_file = fullfile(folder, "device1.mat");
Load_Data = load(load_file).Right_Data; % Device 3 data
whos
plot(Load_Data(:,1), Load_Data(:,2:end));