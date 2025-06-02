load("..\Readings\2024-12-05_18-15" + "\device1.mat");
load("..\Readings\2024-12-05_18-15" + "\device3.mat");
plot(device1.Right_Data(:,1), device1.Right_Data(:,2:end));