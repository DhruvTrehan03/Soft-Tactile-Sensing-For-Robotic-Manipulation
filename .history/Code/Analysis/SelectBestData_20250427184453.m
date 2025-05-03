baseFolder = "C:\Users\dhruv\Soft-Tactile-Sensing-For-Robotic-Manipulation\Readings";
diameters = ["10mm", "20mm", "30mm", "40mm"];
outputFolder = "C:\Users\dhruv\Soft-Tactile-Sensing-For-Robotic-Manipulation\Code\SavedVariables\BestData";

if ~isfolder(outputFolder)
    mkdir(outputFolder);
end

for d = 1:length(diameters)
    diameterFolder = fullfile(baseFolder, diameters(d));
    subfolders = dir(diameterFolder);
    subfolders = subfolders([subfolders.isdir] & ~startsWith({subfolders.name}, '.')); % Ignore '.' and '..'

    fprintf("Processing %s...\n", diameters(d));
    bestData = [];
    bestSubfolder = "";

    for s = 1:length(subfolders)
        subfolderPath = fullfile(diameterFolder, subfolders(s).name);
        dataFiles = dir(fullfile(subfolderPath, "*.mat"));

        for f = 1:length(dataFiles)
            dataPath = fullfile(subfolderPath, dataFiles(f).name);
            data = load(dataPath);

            % Assuming the variable of interest is named 'plotthis_right' or similar
            if isfield(data, 'plotthis_right')
                plotData = data.plotthis_right;
            elseif isfield(data, 'plotthis_truth')
                plotData = data.plotthis_truth;
            else
                continue; % Skip if no relevant data
            end

            figure;
            plot(plotData(:, 1), plotData(:, 2));
            title(sprintf("Diameter: %s, Subfolder: %s, File: %s", diameters(d), subfolders(s).name, dataFiles(f).name));
            xlabel("Time");
            ylabel("Amplitude");
            grid on;

            choice = input("Is this the best data? (y/n): ", 's');
            close;

            if lower(choice) == 'y'
                bestData = plotData;
                bestSubfolder = subfolders(s).name;
                break;
            end
        end

        if ~isempty(bestData)
            break;
        end
    end

    if ~isempty(bestData)
        save(fullfile(outputFolder, sprintf("BestData_%s.mat", diameters(d))), 'bestData', 'bestSubfolder');
        fprintf("Best data for %s saved from subfolder: %s\n", diameters(d), bestSubfolder);
    else
        fprintf("No best data selected for %s.\n", diameters(d));
    end
end
