clear
clc

% Prompt user to select devices
init_left = input('Initialize Left (COM13)? (1 for yes, 0 for no): ');
init_right = input('Initialize Right (COM14)? (1 for yes, 0 for no): ');
init_truth = input('Initialize Truth (COM15)? (1 for yes, 0 for no): ');

% Create parallel pool if not already open
poolobj = gcp('nocreate');
if isempty(poolobj)
    parpool; % Start parallel pool
end

% Get the number of readings
n = input("What value of n (readings)? ");

% Get the current time for unique filenames
current_time = datetime('now', 'Format', 'yyyy-MM-dd_HH-mm-ss');
time_str = char(current_time);
save_dir = fullfile('C:\Users\dhruv\Soft-Tactile-Sensing-For-Robotic-Manipulation\Readings\40mm\', time_str);
mkdir(save_dir)

% Create futures for each device
futures = [];

if init_left
    futures(end+1) = parfeval(@readSerialData, 1, "COM13", save_dir, 'left', n);
end
if init_right
    futures(end+1) = parfeval(@readSerialData, 1, "COM14", save_dir, 'right', n);
end
if init_truth
    futures(end+1) = parfeval(@readSerialData, 1, "COM15", save_dir, 'truth', n);
end

% Wait for all tasks to complete
wait(futures);

disp('All readings completed and saved.');

%% Function to Read Serial Data in Parallel
function readSerialData(port, save_dir, label, n)
    device = serialport(port, 115200);
    device.Timeout = 25;
    write(device, "y", "string"); % Send initialization command
    pause(1); % Allow time for response

    plotthis = [];

    for i = 1:n
        current_time = datenum(datetime('now', 'Format', 'HH:mm:ss.SSS'));
        try
            data = str2num(readline(device)); % Read data
            if ~isempty(data)
                plotthis = [plotthis; current_time, data];
            end
        catch
            warning("Error reading from %s", port);
        end
        pause(0.01); % Small pause to allow different rates
    end

    % Save data
    save(fullfile(save_dir, [label, '.mat']), 'plotthis');
    clear device;
end
