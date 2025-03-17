clear
clc

% Initialize Right Device
port = "COM14";
device = serialport(port, 115200);
device.Timeout = 25;
write(device, "y", "string"); % Send initialization command

% Get unique save folder based on date and minute
current_time = datetime('now', 'Format', 'yyyy-MM-dd_HH-mm');
time_str = char(current_time);
save_dir = fullfile('C:\Users\dhruv\Soft-Tactile-Sensing-For-Robotic-Manipulation\Readings\40mm\Right', time_str);
mkdir(save_dir)

% Get number of readings
n = 200;

% Data logging
plotthis = [];
for i = 1:n
    current_time = datenum(datetime('now', 'Format', 'HH:mm:ss.SSS'));
    try
        data = str2num(readline(device));
        if ~isempty(data)
            plotthis = [plotthis; current_time, data];
        end
    catch
        warning("Error reading from %s", port);
    end
    pause(0.01); % Small pause
    disp(i)
end

% Save data
save(fullfile(save_dir, 'right.mat'), 'plotthis');
clear device;
disp('Right device readings saved.');
