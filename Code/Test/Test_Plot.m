clear;

% Initialize devices with different data rates
Left = serialport("COM13", 115200);
Left.Timeout = 1;

Right = serialport("COM14", 115200);
Right.Timeout = 1;

Load = serialport("COM15", 115200);
Load.Timeout = 1;

% Command devices to start (if required)
Left.write("y", "string");
Right.write("y", "string");
Load.write("y", "string");


% Specify the duration of data collection
collection_duration = 4620; % in seconds, as a rule of thumb 45*number of twists + 20
start_time = datetime('now');
disp(datestr(start_time, 'yyyy-mm-dd_HH-MM'))
save_dir = fullfile('C:\Users\dhruv\4th Year Project\MATLAB\Large_Data\', datestr(start_time, 'yyyy-mm-dd_HH-MM'));
mkdir(save_dir);

% Buffers to store data
Right_Data = [];
Left_Data = [];
Load_Data = [];

% Continue reading until the collection duration is reached
while seconds(datetime('now') - start_time) < collection_duration
    disp(seconds(datetime('now') - start_time))
    % Read from device1
    if Left.NumBytesAvailable > 0
        line = readline(Left);
        Right_Data = [Right_Data; datenum(datetime('now')), str2num(line)]; %#ok<*ST2NM>
    end
    
    % Read from device2
    if Right.NumBytesAvailable > 0
        line = readline(Right);
        Left_Data = [Left_Data; datenum(datetime('now')), str2num(line)];
    end
    
    % Read from device3
    if Load.NumBytesAvailable > 0
        line = readline(Load);
        Load_Data = [Load_Data; datenum(datetime('now')), str2num(line)];
    end
end

% Save collected data


save(fullfile(save_dir, 'device1.mat'), 'Right_Data');
save(fullfile(save_dir, 'device2.mat'), 'Left_Data');
save(fullfile(save_dir, 'device3.mat'), 'Load_Data');

% Close serial ports
clear Left Right Load;

% Post-processing can now align data streams by timestamps if required.
