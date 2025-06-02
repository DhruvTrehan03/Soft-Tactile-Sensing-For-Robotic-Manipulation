load("..\Readings\2024-12-05_18-15" + "\device1.mat");
load("..\Readings\2024-12-05_18-15" + "\device3.mat");

plot(Load_Data(:,1), Load_Data(:,2:end));

%find peaks list 
[pks, locs] = findpeaks(Load_Data(:,2), 'MinPeakHeight', 0.005);
disp(pks);