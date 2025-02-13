clear; clc; close all;

% Material and Roller Properties
E  = 5e6;      % Young's Modulus (Pa)
nu = 0.45;     % Poisson’s Ratio
E_star = E / (1 - nu^2); % Effective modulus

R = 1;      % Roller radius (m)
F_n = 1;       % Normal force (N)
mu_ro = 0.5;  % Rolling friction coefficient
lambda_ro = 1; % Rolling resistance length scale (m)

% Hertzian Contact Parameters
delta = (F_n^(2/3)) / (E_star * R^(1/3));  % Indentation depth
a = ((3 * F_n * R) / (4 * E_star))^(1/3);  % Contact half-width

% Define Surface Positions
x = linspace(-2*a, 2*a, 1000);

% Compute Hertzian Displacement
u_hertz = delta * (1 - (x.^2 / a^2));

% Compute Rolling Resistance Contribution (Corrected)
u_roll = (mu_ro * lambda_ro * F_n) / (E_star * a);

% Total Displacement
u_total = u_hertz + u_roll;

% Plot Results
figure;
plot(x * 1e3, u_hertz * 1e6, 'b-', 'LineWidth', 2); hold on;
plot(x * 1e3, u_total * 1e6, 'r--', 'LineWidth', 2);
legend('Hertzian Contact', 'With Rolling Resistance');
xlabel('Position (mm)');
ylabel('Displacement (µm)');
title('Surface Displacement Under a Rolling Object');
grid on;
