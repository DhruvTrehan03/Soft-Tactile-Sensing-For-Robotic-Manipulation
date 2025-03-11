clear;
run('Source/eidors-v3.11-ng/eidors/eidors_startup.m');  % Initialize EIDORS
mdl = stl_read("C:\Users\dhruv\Documents\Tripos\SoftTactileSensingForRoboticManipulation\Code\Test\Shiny Jarv-Luulia.stl");
show_fem(mdl)