baseFolder = "C:\Users\dhruv\Soft-Tactile-Sensing-For-Robotic-Manipulation\Readings";
diameters = ["10mm", "20mm", "30mm", "40mm"];
outputFolder = "C:\Users\dhruv\Soft-Tactile-Sensing-For-Robotic-Manipulation\Code\SavedVariables\BestData";

load('C:\Users\dhruv\Soft-Tactile-Sensing-For-Robotic-Manipulation\Readings\40mm\2025-03-18_11-15\20mm_right.mat')
load('C:\Users\dhruv\Soft-Tactile-Sensing-For-Robotic-Manipulation\Readings\40mm\2025-03-18_11-15\20mm_truth.mat') 

plot(plotthis_right(:,1),plotthis_right(:,2))
figure();
plot(plotthis_truth(:,1),plotthis_truth(:,2))  