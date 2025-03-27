clear
clc

% Initialize COM15 (fast) and optionally COM14 (slow)
init_truth = 1; % Always read from COM15
init_right = input('Initialize Right (COM14)? (1 for yes, 0 for no): ');

% Open Serial Ports
truth = serialport("COM15", 115200);
truth.Timeout = 1; % Short timeout to avoid blocking
if init_right
    right = serialport("COM14", 115200);
    right.Timeout = 1; % Short timeout to prevent slowdowns
end

% Send initialization command
write(truth, "y", "string");
if init_right, write(right, "y", "string"); end

% Get the current time to create unique file names (minute-based)
current_time = datetime('now', 'Format', 'yyyy-MM-dd_HH-mm');
time_str = char(current_time);
save_dir = fullfile('C:\Users\dhruv\Soft-Tactile-Sensing-For-Robotic-Manipulation\Readings\40mm\', time_str);
mkdir(save_dir)

% Get total runtime in seconds instead of number of readings
runtime = input("How many seconds to collect data? ");
tic; % Start timing

% Initialize data storage
plotthis_truth = [];
plotthis_right = [];

% Start reading loop
while toc < runtime  % Run until the set time is reached
    disp(toc)
    current_time = datenum(datetime('now', 'Format', 'HH:mm:ss.SSS'));
    
    % Read data from COM15 (always)
    try
        data_truth = str2num(readline(truth));
        plotthis_truth = [plotthis_truth; current_time, data_truth]; % Store truth data
    catch
        warning("Error reading from COM15");
    end
    
    % Try reading from COM14 (only if new data is available)
    if init_right && right.NumBytesAvailable > 0
        try
            data_right = str2num(readline(right));
            plotthis_right = [plotthis_right; current_time, data_right]; % Store only new right data
        catch
            warning("Error reading from COM14");
        end
    end
    
    % Small pause to avoid overwhelming MATLAB
    pause(0.001);
end

% Save data
save(fullfile(save_dir, '10mm_truth.mat'), 'plotthis_truth');
if init_right && ~isempty(plotthis_right)
    save(fullfile(save_dir, '10mm_right.mat'), 'plotthis_right');
end

% Close serial ports
clear truth;
if init_right, clear right; end

disp('Data collection complete.');
