load('maxima_data.mat');
whos
scatter3(maxima_k, maxima_sigma, torque_values, 20, maxima_corr, 'filled');