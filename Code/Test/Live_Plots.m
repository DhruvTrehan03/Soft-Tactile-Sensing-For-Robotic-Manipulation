% For debugging: see EIT board data in real time

clear device
device = serialport("COM15",115200);
device.Timeout = 25;

device.write("y", "string");

for i=1:1
    data = readline(device); 
    i;
end

plotthis = str2num(data);

n = 300;
for i = 1:n
    i;
    data = readline(device);
    disp(data)
    if ~isempty(data)
        data = str2num(data);
        plotthis = [plotthis; data];
        clf;
        plot(data); % this line
        % Assumes that ranking variable already exists from a previous script...
        % plot(plotthis(:, ranking(1:200)), 'linewidth', 2);
        % plot(plotthis, 'linewidth', 2); % this line
        % plot(mean(plotthis(:, 1:100).'));
        % hold on
        % plot(mean(plotthis(:, 101:200).'));

        set(gca, 'color', 'w', 'linewidth', 2, 'fontsize', 15);
        set(gcf, 'color', 'w');
        box off
        ylabel("Magnitude");
        ylim([0 1.2]);
        xlim([0 n]);
        drawnow();
    end
end

clear device

