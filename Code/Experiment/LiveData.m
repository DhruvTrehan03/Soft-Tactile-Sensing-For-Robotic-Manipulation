% For debugging: see EIT board data in real time

clear

device1 = serialport("COM13", 115200);
device1.Timeout = 25;

device2 = serialport("COM14", 115200);
device2.Timeout = 25;

device3 = serialport("COM15", 115200);
device3.Timeout = 25;

device1.write("y", "string");
device2.write("y", "string");

% Get the current time to create unique file names
current_time = datetime('now', 'Format', 'yyyyMMdd_HHmmss');  % Current time with format YYYYMMDD_HHMMSS

% Convert the datetime object to a string for the file name
time_str = char(current_time);

% Specify the directory to save the data
save_dir = ['C:\Users\dhruv\4th Year Project\MATLAB\Large_Data\',time_str];
mkdir(save_dir)


% Initialize data buffers
for i=1:1
    current_time = datenum(datetime('now', 'Format', 'HH:mm:ss.SSS'));
    data1 = str2num(readline(device1));
    data2 = str2num(readline(device2));
    data3 = str2num(readline(device3));
    i;
end

plotthis1 = [current_time,data1];
plotthis2 = [current_time,data2];
plotthis3 = [current_time,data3];
% Number of readings to capture
n = 150;

for i = 1:n
    disp(i);
    % Read data from devices
    data1 = str2num(readline(device1));
    data2 = str2num(readline(device2));
    data3 = str2num(readline(device3));
    
    % Get the current time for this reading
    
    
    % Process data if not empty and append time at the same time
    if ~isempty(data1)
        current_time = datenum(datetime('now', 'Format', 'HH:mm:ss.SSS'));
        data1 = [current_time,data1];
        plotthis1 = [plotthis1; data1]; % Append new data to plotthis1
    end
    if ~isempty(data2)
        current_time = datenum(datetime('now', 'Format', 'HH:mm:ss.SSS'));
        data2 = [current_time,data2];
        plotthis2 = [plotthis2; data2]; % Append new data to plotthis2
    end
    if ~isempty(data3)
        current_time = datenum(datetime('now', 'Format', 'HH:mm:ss.SSS'));
        data3 = [current_time,data3]; 
        plotthis3 = [plotthis3; data3]; % Append new data to plotthis3 
    end
    
    % Optionally, save time_log for debugging
    time_log(i) = current_time; % Store time for each reading
end


% Add the time_log as the first column of each plotthis data and save

save(fullfile(save_dir, ['left.mat']), 'plotthis1');

save(fullfile(save_dir, ['right.mat']), 'plotthis2');

save(fullfile(save_dir, ['load.mat']), 'plotthis3');

% Clear devices
clear device1, clear device2, clear device3
