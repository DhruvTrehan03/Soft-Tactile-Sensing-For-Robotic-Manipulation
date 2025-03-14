clear; clc;

% Initialize the serial port for COM14 (Right Device)
right = serialport("COM14", 115200);
right.Timeout = 25; % Set a timeout to avoid indefinite waiting

% Send initialization command
write(right, "y", "string");

% Specify the number of readings
% n = input("Enter the number of readings to collect: ");
n=11;

% Buffers for first 10 readings (Calibration)
calibration_data = [];

% Buffers for storing scaled data
plotthis_right = zeros(1,1024);

% Setup live plot
figure;
hold on;
grid on;
xlabel('Reading Number');
ylabel('Scaled Data');
title('Live Data from COM14');
h = animatedline('Color', 'b', 'LineWidth', 1.5);

for i = 1:n
        % Read and scale the data
            raw_data = str2num(readline(right)); %#ok<ST2NM>
            
            if ~isempty(raw_data) 
                % Store the scaled data
                plotthis_right = [plotthis_right; raw_data];
                disp(i);
                % Update live plot
                addpoints(h, i, mean(raw_data));
                drawnow;
            end

    % end
end
plotthis_right = plotthis_right(2:end,:);
plot(plotthis_right)
% Save data
save_dir = 'C:\Users\dhruv\Soft-Tactile-Sensing-For-Robotic-Manipulation\Readings\Funky';
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end
save(fullfile(save_dir, 'Funky_Heart_3_2_3.mat'), 'plotthis_right');

% Close the serial port safely
clear right;

disp('Data collection complete.');
