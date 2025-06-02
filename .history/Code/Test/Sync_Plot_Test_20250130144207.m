% Set up serial ports for three devices
clear device1 device2 device3;
device1 = serialport("COM13", 115200);
device2 = serialport("COM14", 115200);
device3 = serialport("COM15", 115200);

% Set a timeout for each device
device1.Timeout = 25;
device2.Timeout = 25;
device3.Timeout = 25;

% Example setup for data synchronization
timestamps1 = [];
timestamps2 = [];
timestamps3 = [];

data1 = [];
data2 = [];
data3 = [];

% Set up plots
figure;
h1 = subplot(3, 1, 1);
p1 = plot(nan, nan, 'linewidth', 2);
title('Device 1 Data');
xlabel('Time (s)');
ylabel('Magnitude');

h2 = subplot(3, 1, 2);
p2 = plot(nan, nan, 'linewidth', 2);
title('Device 2 Data');
xlabel('Time (s)');
ylabel('Magnitude');

h3 = subplot(3, 1, 3);
p3 = plot(nan, nan, 'linewidth', 2);
title('Device 3 Data');
xlabel('Time (s)');
ylabel('Magnitude');

% Initialize timing
tic; % Start a timer for relative timestamps

while true
    % Read data from devices
    if device1.NumBytesAvailable > 0
        line1 = readline(device1);
        [t1, d1] = processData(line1, toc); % Include timestamp
        timestamps1 = [timestamps1; t1];
        data1 = [data1; d1];
    end

    if device2.NumBytesAvailable > 0
        line2 = readline(device2);
        [t2, d2] = processData(line2, toc); % Include timestamp
        timestamps2 = [timestamps2; t2];
        data2 = [data2; d2];
    end

    if device3.NumBytesAvailable > 0
        line3 = readline(device3);
        [t3, d3] = processData(line3, toc); % Include timestamp
        timestamps3 = [timestamps3; t3];
        data3 = [data3; d3];
    end

    % Align data using timestamps (simplified example)
    minTime = max([min(timestamps1), min(timestamps2), min(timestamps3)]);
    maxTime = min([max(timestamps1), max(timestamps2), max(timestamps3)]);
    
    % Filter data to aligned time range
    idx1 = timestamps1 >= minTime & timestamps1 <= maxTime;
    idx2 = timestamps2 >= minTime & timestamps2 <= maxTime;
    idx3 = timestamps3 >= minTime & timestamps3 <= maxTime;
    
    alignedData1 = data1(idx1);
    alignedData2 = data2(idx2);
    alignedData3 = data3(idx3);
    
    % Update plots
    set(p1, 'XData', timestamps1(idx1), 'YData', alignedData1);
    set(p2, 'XData', timestamps2(idx2), 'YData', alignedData2);
    set(p3, 'XData', timestamps3(idx3), 'YData', alignedData3);
    
    drawnow;
end

% Function to process incoming data and add timestamps
function [timestamp, data] = processData(line, elapsedTime)
    % Example: Split a comma-separated line into a timestamp and data value
    splitLine = str2double(strsplit(line, ','));
    timestamp = elapsedTime; % Use elapsed time as timestamp
    data = splitLine(1); % Assume first value is data
end
