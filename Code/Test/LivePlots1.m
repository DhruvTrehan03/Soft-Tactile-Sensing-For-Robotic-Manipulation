% For debugging: see EIT board data in real time

clear device1, clear device2
device1 = serialport("COM13",115200);
device1.Timeout = 25;

device2 = serialport("COM14",115200);
device2.Timeout = 25;

device1.write("y", "string");
device2.write("y", "string");

for i=1:1
    data1 = readline(device1);
    data2 = readline(device2);
    i
end

plotthis1 = str2num(data1);
plotthis2 = str2num(data2);

n = 100;
for i = 1:n
    i
    data1 = readline(device1);
    data2 = readline(device2);
    if ~isempty(data1)
        data1 = str2num(data1);
        plotthis1 = [plotthis1; data1];
        data2 = str2num(data2);
        plotthis2 = [plotthis2; data2];
        clf;
        % plot(data); % this line
        % Assumes that ranking variable already exists from a previous script...
        % plot(plotthis(:, ranking(1:200)), 'linewidth', 2);
        % h1= subplot(221);
        % plot(plotthis1(:,1:50), 'linewidth', 2);
        heatmap(normalize(plotthis1, "range", [0 1]).', "colormap", hot); grid off;
        % h1.YDisplayLabels = [];  % This removes the y-axis labels from the heatmap
        % set(gca, 'color', 'w', 'linewidth', 2, 'fontsize', 15);
        % set(gcf, 'color', 'w');
        % box off
        % ylabel("Magnitude");
        % ylim([0 1.2]);
        % xlim([0 n]);
        drawnow();
        % h2= subplot(222);
        % plot(plotthis2(:,:), 'linewidth', 2); % this line
        % % plot(mean(plotthis(:, 1:100).'));
        % % hold on
        % % plot(mean(plotthis(:, 101:200).'));
        % 
        % set(gca, 'color', 'w', 'linewidth', 2, 'fontsize', 15);
        % set(gcf, 'color', 'w');
        % box off
        % ylabel("Magnitude");
        % ylim([0 1.2]);
        % xlim([0 n]);
        % drawnow();
    end
end

clear device1, clear device2, clear device3

