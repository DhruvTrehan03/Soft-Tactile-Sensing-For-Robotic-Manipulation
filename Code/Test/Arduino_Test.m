% For debugging: see EIT board data in real time

clear device1, clear device2
figure();
device1 = serialport("COM15",115200);
device1.Timeout = 25;

% device2 = serialport("COM14",115200);
% device2.Timeout = 25;

device1.write("y", "string");
% device2.write("y", "string");

for i = 1:1
    data1 = readline(device1);
    % data2 = readline(device2);
    i
end

plotthis1 = str2num(data1);
% plotthis2 = str2num(data2);

n = 120;
for i = 1:n
    i
    data1 = readline(device1);
    % data2 = readline(device2);
    if ~isempty(data1)
        data1 = str2num(data1);
        plotthis1 = [plotthis1; data1];
        % data2 = str2num(data2);
        % plotthis2 = [plotthis2; data2];
        clf;
        
        % Heatmap - Remove y-axis tick labels
        
        plot(plotthis1)
        % Remove y-axis tick labels by modifying the axes
        grid off;
        drawnow();
        
        % Line plot - Remove y-axis tick labels
        % h2 = subplot(222);
        % plot(plotthis2(:, 1:20), 'linewidth', 2);  % this line
        % set(gca, 'YTickLabel', []);  % This removes the y-axis labels for the line plot
        % set(gca, 'color', 'w', 'linewidth', 2, 'fontsize', 15);
        % set(gcf, 'color', 'w');
        % box off;
        % ylabel("Magnitude");
        % ylim([0 1.2]);
        % xlim([0 n]);
        % drawnow();
    end
end

clear device1, clear device2, clear device3
