clear;
clc;

% Initialize the serial port for a single device
device = serialport("COM14", 115200);
device.Timeout = 1;

% Command device to start (if required)
device.write("y", "string");

% Specify the duration of data collection (in seconds)
collection_duration = 4620;  
start_time = datetime('now');

% Set up figure for live plotting
figure;
hold on;
xlabel('Time (s)');
ylabel('Sensor Value');
title('Live Data Plot');
grid on;

% Initialize data storage
time_stamps = [];
data_values = [];

% Start data collection loop
while seconds(datetime('now') - start_time) < collection_duration
    elapsed_time = seconds(datetime('now') - start_time);
    disp(elapsed_time);
    
    % Read data if available
    if device.NumBytesAvailable > 0
        line = readline(device);
        data = str2num(line); %#ok<ST2NM>
        
        if ~isempty(data)
            % Store data
            time_stamps = [time_stamps; elapsed_time];
            data_values = [data_values; data];
            
            % Update live plot
            plot(time_stamps, data_values, 'b.-');
            drawnow;
        end
    end
end

% Save collected data
save('device_data.mat', 'time_stamps', 'data_values');

% Close the serial port
clear device;

disp('Data collection complete.');
