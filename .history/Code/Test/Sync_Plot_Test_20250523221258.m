folder = "..\Readings\2024-12-05_18-15"
load_file = fullfile(folder, "device3.mat");
Load_Data = load(load_file).Load_Data; % Device 3 data
whos
disp(Load_Data)