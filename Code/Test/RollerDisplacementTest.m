% MATLAB Simulation: Surface Displacement Under a Rolling Cylinder

clear; clc; close all;

% Define Material Properties
E  = 5e6;    % Young's Modulus (Pa) [Example: Soft Polymer]
nu = 0.45;   % Poisson's Ratio
E_star = E / (1 - nu^2); % Effective modulus

% Define Roller Properties
R = 0.01;    % Radius of the Roller (m) [Example: 10 mm]
F_n = 1;     % Normal Load (N)
mu_ro = 0.01; % Rolling friction coefficient
lambda_ro = 1e-3; % Rolling resistance length scale (m)

% Compute Hertzian Contact Parameters
delta = (9/16)^(1/3) *(F_n^(2/3)) / ((E_star)^(2/3) * R^(1/3));  % Indentation depth
a = ((3 * F_n * R) / (4 * E_star))^(1/3);  % Contact half-width

% Define x-axis (surface positions)
x = linspace(-2*a, 2*a, 1000); % Simulation domain

% Compute Surface Displacement Using Hertz Theory
u_hertz = -delta + (x.^2)/(2*R); % Parabolic profile

% Compute Rolling Resistance Term
u_roll = (mu_ro * lambda_ro * F_n) / (E_star * a);

% Compute Total Displacement
u_total = u_hertz ;

% Plot Results
figure;
plot(x * 1e3, u_hertz * 1e6, 'b-', 'LineWidth', 2); hold on;
plot(x * 1e3, u_total * 1e6, 'r--', 'LineWidth', 2);
legend('Hertzian Contact', 'With Rolling Resistance');
xlabel('Position (mm)');
ylabel('Displacement (µm)');
title('Surface Deformation Under a Rolling Cylinder');
grid on;
