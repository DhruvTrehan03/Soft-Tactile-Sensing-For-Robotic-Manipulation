% For debugging: see EIT board data in real time

clear device1, clear device2, clear device3
device1 = serialport("COM13",115200);
device1.Timeout = 25;

device2 = serialport("COM14",115200);
device2.Timeout = 25;

device3 = serialport("COM15",115200);
device3.Timeout = 25;

device1.write("y", "string");
device2.write("y", "string");

for i=1:1
    data1 = str2num(readline(device1));
    data2 = str2num(readline(device2));
    data3 = str2num(readline(device3));
    i
end

n = 50;

plotthis1 = nan(n,length(data1));
plotthis2 = nan(n,length(data2));
plotthis3 = nan(n,length(data3));


plotthis1(1,:) = data1;
plotthis2(1,:) = data2;
plotthis3(1,:) = data3;


for i = 1:n
    i
    data1 = readline(device1);
    data2 = readline(device2);
    data3 = readline(device3);
    if ~isempty(data1)
        if ~isempty(data2)
            if ~isempty(data3)
        data1 = str2num(data1);
        data2 = str2num(data2);
        data3 = str2num(data3);

        plotthis1(i,:) = data1;
        plotthis2(i,:) = data2;
        plotthis3(i) = data3;
        
        % plot(data); % this line
        % Assumes that ranking variable already exists from a previous script...
        % plot(plotthis(:, ranking(1:200)), 'linewidth', 2);
        
        h1= subplot(221);
        % heatmap(normalize(plotthis1, "range", [0 1]).', "colormap", hot);
        plot(plotthis1(:,1:20));
        grid off
        drawnow;
        
        h2 = subplot(2,2,[3,4]);
        plot(plotthis3);
        
        % h2 = subplot(212);
        % plot(plotthis1(:,1:20), 'linewidth', 2);
        % set(gca, 'color', 'w', 'linewidth', 2, 'fontsize', 15);
        % set(gcf, 'color', 'w');
        % box off
        % ylabel("Magnitude");
        % ylim([0 1.2]);
        % xlim([0 n]);
        % drawnow();
        % figure;
        
        h3 = subplot(222);
        heatmap(normalize(plotthis2, "range", [0 1]).', "colormap", hot); grid off;
        % plot(plotthis2(:,1:20));
        drawnow;
         
        % h4= subplot(214);
        % plot(plotthis2(:,1:20), 'linewidth', 2); % this line
        % % plot(mean(plotthis(:, 1:100).'));
        % % hold on
        % % plot(mean(plotthis(:, 101:200).'));
         
        % set(gca, 'color', 'w', 'linewidth', 2, 'fontsize', 15);
        % set(gcf, 'color', 'w');
        % box off
        % ylabel("Magnitude");
        % ylim([0 1.2]);
        % xlim([0 n]);
        % drawnow();
            end
        end
    end
end

clear device1,clear device2, clear device3

