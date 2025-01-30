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
    data1 = readline(device1);
    data2 = readline(device2);
    data3 = readline(device3);
    i
end

plotthis1 = str2num(data1);
plotthis2 = str2num(data2);
plotthis3 = str2num(data3);

n = 10;

heatmapData = nan(1024, n);
% h1= subplot(211);
% heatmapPlot = imagesc(heatmapData, [0 1]); % Initialize heatmap with range [0, 1]
% colormap(hot);
% colorbar;
% xlabel('Time (n)');
% ylabel('Sensor/Data Points');
% title('Dynamic Heatmap');

loadcellData = nan(1,n);
h2 = subplot(212);
loadcellPlot = plot(loadcellData);



for i = 1:n
    i
    data1 = readline(device1);
    data2 = readline(device2);
    data3 = readline(device3);
    
    data1 = str2num(data1);
    data2 = str2num(data2);
    data3 = str2num(data3);

    % plotthis1 = [plotthis1; data1];
    plotthis2 = [plotthis2; data2];
    
    h1 = subplot(211);
    

    loadcellData(:, i) = data3;
    currentData = data1; % Normalize new data
    
    % Add the current data as a column to the heatmap matrix
    heatmapData(:, i) = data1'; % Transpose to match orientation
    heatmapDataNorm = normalize(heatmapData, "range", [0,1]);
    heatmap(heatmapDataNorm, 'colormap', hot);
    % Update the heatmap
    
    set(loadcellPlot, 'YData', loadcellData);
    grid off
    drawnow;
    
   
end
figure;
normalizedFullData = normalize(heatmapData, "range", [0, 1]);  % Normalize the full data set
imagesc(normalizedFullData);
colormap(hot);
colorbar;
% Create a new figure to display the full normalized heatmap

testData = rand(10, 5);
% h3 = heatmap(normalizedFullData, 'colormap', hot);
xlabel('Time (n)');
ylabel('Sensor/Data Points');
title('Normalized Heatmap Over Full Dataset');
clear device1,clear device2, clear device3
