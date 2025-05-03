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
    bestSubfolder = "";

    for s = 1:length(subfolders)
        subfolderPath = fullfile(diameterFolder, subfolders(s).name);
        dataFiles = dir(fullfile(subfolderPath, "*.mat"));

        % Find paired files for the current diameter
        rightFile = "";
        truthFile = "";
        for f = 1:length(dataFiles)
            if contains(dataFiles(f).name, sprintf("%s_right", diameters(d)))
                rightFile = fullfile(subfolderPath, dataFiles(f).name);
            elseif contains(dataFiles(f).name, sprintf("%s_truth", diameters(d)))
                truthFile = fullfile(subfolderPath, dataFiles(f).name);
            end
        end

        % Skip if the pair is incomplete
        if isempty(rightFile) || isempty(truthFile)
            continue;
        end

        % Load and plot the data for inspection
        dataRight = load(rightFile);
        dataTruth = load(truthFile);

        if isfield(dataRight, 'plotthis_right') && isfield(dataTruth, 'plotthis_truth')
            plotDataRight = dataRight.plotthis_right;
            plotDataTruth = dataTruth.plotthis_truth;

            figure;
            subplot(2, 1, 1);
            plot(plotDataRight(:, :), plotDataRight(:, 2));
            title(sprintf("Right Data - Diameter: %s, Subfolder: %s", diameters(d), subfolders(s).name));
            xlabel("Time");
            ylabel("Amplitude");
            grid on;

            subplot(2, 1, 2);
            plot(plotDataTruth(:, 1), plotDataTruth(:, 2));
            title(sprintf("Truth Data - Diameter: %s, Subfolder: %s", diameters(d), subfolders(s).name));
            xlabel("Time");
            ylabel("Amplitude");
            grid on;

            choice = input("Is this the best data? (y/n): ", 's');
            close;

            if lower(choice) == 'y'
                bestSubfolder = subfolders(s).name;

                % Move the selected pair of files to the output folder
                movefile(rightFile, fullfile(outputFolder, sprintf("%s_right.mat", diameters(d))));
                movefile(truthFile, fullfile(outputFolder, sprintf("%s_truth.mat", diameters(d))));
                fprintf("Best data for %s moved from subfolder: %s\n", diameters(d), bestSubfolder);
                break;
            end
        end
    end

    if isempty(bestSubfolder)
        fprintf("No best data selected for %s.\n", diameters(d));
    end
end
