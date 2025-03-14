clear;
load('C:\Users\dhruv\Soft-Tactile-Sensing-For-Robotic-Manipulation\Readings\2024-12-05_18-15\device2.mat')
load('C:\Users\dhruv\Soft-Tactile-Sensing-For-Robotic-Manipulation\Readings\2024-12-05_18-15\device3.mat')

burn_in = 0.0001;
end_proportion =  0.75;
resolution = 5000;
EIT = Left_Data(:,2:end);
EIT_Time = Left_Data(:,1);
[pks,locs] = findpeaks(Load_Data(:,2),'MinPeakHeight',0.005);
torqueTimes = Load_Data(locs,1);

closest_values = interp1(EIT_Time, 1:length(EIT_Time), torqueTimes, 'next', 'extrap');
    
% Convert NaN values (out-of-range) to valid indices
closest_values(isnan(closest_values)) = length(EIT_Time);

% Convert to integer indices
closest_values = ceil(closest_values);

figure();hold on;
plot(EIT(closest_values(1),:) - EIT(100,:)) ;
hold off;
hom = EIT(100,:),0,0.45;
trainTorquePeaks = torquePeaks(1:9);
testTorquePeaks = torquePeaks(10:99);
save("SavedVariables\TorqueFitting\Torque.mat","trainTorquePeaks","testTorquePeaks")
for i = 1:9
    data = EIT(closest_values(i),:);
    % plot(data)
    data_diff = abs(data-hom);
    % plot(data_diff)
    save(sprintf("SavedVariables\\TorqueFitting\\Train_%i",i),"data_diff");
    disp(i);
end
for i=10:99
    data = clip(EIT(closest_values(i),:),0,0.45);
    data_diff = data-hom;
    save(sprintf("SavedVariables\\TorqueFitting\\Test_%i",i),"data_diff");
    disp(i);
end 

%%

function [Data, Time, retained_indices] = preprocess(Raw_Data, burn_in, end_proportion)
    % --- Preprocess Raw Data ---
    % Extract start and end indices based on burn-in and end proportion
    start_time = ceil(burn_in * size(Raw_Data, 1));
    % disp(start_time)
    end_time = ceil(end_proportion * size(Raw_Data, 1));
    
    % Trim the raw data
    Raw_Data = Raw_Data(start_time:end_time, :);
    
    % Extract Time and Data
    Time = Raw_Data(:, 1);
    Data = Raw_Data(:, 2:end);
    
    % Remove columns with all NaN values
    columns_with_zeros = ~all(Data==0);  % Identify columns that are entirely NaN
    Data = Data(:, columns_with_zeros);       % Remove those columns
    retained_indices = find(~columns_with_zeros);

    % % Center and normalize the data
    % Data = Data - mean(Data,2); % Center each channel
    % Data = Data ./ std(Data,1,2); % Normalize each channel

    Data = zscore(Data,0,1);
end

function [Torque_Interp, EIT_Interp, dense_time] = interpolate_data(Torque_Time, Torque, EIT_Time, EIT, resolution,test)
    % This function interpolates both Torque and EIT data to a new dense time grid.
    %
    % Inputs:
    %   Torque_Time: Time vector for the Torque data
    %   Torque: Torque data (size [t1, n])
    %   EIT_Time: Time vector for the EIT data
    %   EIT: EIT data (size [t2, m])
    %   resolution: Desired number of time points for the dense time grid (scalar)
    %
    % Outputs:
    %   Torque_Interp: Interpolated Torque data
    %   EIT_Interp: Interpolated EIT data
    %   dense_time: The new dense time grid
    
    % --- Step 1: Create Dense Time Grid ---
    % Create the dense time grid with a finer resolution
    start_time = min(Torque_Time(1), EIT_Time(1));  % Start time (from either Torque or EIT)
    end_time = max(Torque_Time(end), EIT_Time(end));  % End time (from either Torque or EIT)
    
    % Create the new dense time grid with the given resolution (number of points)
    dense_time = linspace(start_time, end_time, resolution).';
    
    % --- Step 2: Interpolate Torque Data ---
    Torque_Interp = interp1(Torque_Time, Torque, dense_time, 'linear', 'extrap');
    
    % --- Step 3: Interpolate EIT Data ---
    % Initialize the interpolated EIT data matrix (for each channel)
    EIT_Interp = zeros(length(dense_time), size(EIT, 2));
    % Loop over each channel of EIT data
    for ch = 1:size(EIT, 2)
        EIT_Interp(:, ch) = interp1(EIT_Time, EIT(:, ch), dense_time, 'linear', 'extrap');
    end

    if test == 1
        disp(test)
        % Choose a few channels to compare
        channels_to_plot = [1,10,100];  % Example channels to compare
        for ch = channels_to_plot
            subplot(2,1,1);
            plot(EIT_Time-EIT_Time(1), EIT(:, ch), 'k-', 'LineWidth', 1.5);  % Original EIT data
            subplot(2,1,2);
            plot(dense_time-dense_time(1), EIT_Interp(:, ch), 'r--', 'LineWidth', 1.5);  % Interpolated data
        end
        
        legend('Original EIT Data', 'Interpolated EIT Data');
        xlabel('Time');
        ylabel('Data Value');
        title('EIT Data Interpolation Comparison');
        hold off;
    end


end

function plotHeatmap(data,time)
    num_pairings = size(data, 2);
    
    mask = false(size(time));  % Initialize a mask of the same size as the array
    mask(1:80:end) = true;       % Set every nth position to true
    % Replace all elements not in the nth positions with NaN
    time(~mask) = NaN;         % Replace elements where mask is false with NaN
    % Convert time column to datetime
    time_labels = datetime(time, 'ConvertFrom', 'datenum'); 
    time_labels = datetime(time_labels, "Format", "HH:mm:ss.SSS");
   
    y_labels = 0:num_pairings-1;
    mask = false(size(y_labels));  % Initialize a mask of the same size as the array
    mask([1,end]) = true;       % Set every nth position to true
    y_labels(~mask) = NaN;
    % Normalize the data matrix
    data_normalised = normalize(data, "range", [0 1]).';
    
    % Create the heatmap
    % Generate the heatmap
    h = heatmap(data_normalised,'Colormap', hot, 'ColorbarVisible', 'on');
    h.XDisplayLabels = time_labels;
    h.YDisplayLabels = y_labels;
    % Label the axes and add a title
    xlabel('Time');
    ylabel('Electrode Pairing Index');
    title('Heatmap of Electrode Pairings over Time');
    
    % Adjust grid
    h.GridVisible = 'off';
    
end

function plotLine(data,time)

    % Convert time_log_str to numeric time
    time_str = datetime(time,'ConvertFrom','datenum');  % Convert datetime to numeric for plotting
    
    % Create a new figure for the plot

    hold on;
    
    % Plot each electrode pairing with time on the y-axis and magnitude on the x-axis
    plot(time, data);
 
    
    % Label the axes and add the title
    xlabel('Time');
    ylabel('Torque (Nm)');
    
    % datetick('x', 'HH:MM:SS.SSS');  % Format the x-axis to show time with milliseconds
    title('Electrode Pairings over Time');
    
    % Turn on grid
    grid on;
    
    % Release the hold to allow other plots
    hold off;

end


% Main function to analyze and identify best channels
function [sorted_scores,sorted_indices] = analyze_channels(eit_data, torque_data,time, method) % Choose from: 'correlation', 'snr', 'rms', 'fft', 'dtw'
    % Compute scores for each channel using the selected method
    switch method
        case 'correlation'
            scores = compute_correlation_scores(eit_data, torque_data);
        case 'abscorr'
            scores = compute_abs_correlation_scores(eit_data, torque_data);
        case 'snr'
            scores = compute_snr_scores(eit_data);
        case 'rms'
            scores = compute_rms_scores(eit_data);
        case 'fft'
            scores = compute_fft_similarity(eit_data, torque_data, fs);
        case 'dtw'
            scores = compute_dtw_scores(eit_data, torque_data);
        otherwise
            error('Invalid method selected. Choose from: correlation, snr, rms, fft, dtw');
    end

    % Sort channels based on scores (descending)
    [sorted_scores, sorted_indices] = sort(scores, 'descend');
    
    % Display results
    % disp('Channel Ranking:');
    % for i = 1:num_channels
    %     fprintf('Channel %d: Score = %.4f\n', sorted_indices(i), sorted_scores(i));
    % end
    figure();
    num_subplots = min(20, length(sorted_indices)) + 1;
    for i = 1:num_subplots -1% Plot up to top 3 channels
        % Normalize the EIT signal for plotting
        subplot(num_subplots, 1, i);
        eit_signal_plot = eit_data(:, sorted_indices(i));
        plot(time, eit_signal_plot, 'LineWidth', 1.2,'DisplayName', sprintf('Channel %d', sorted_indices(i)));
        title(sprintf('Channel %d', sorted_indices(i)));
    end
    subplot(num_subplots,1,num_subplots)
    plot(time,torque_data, 'LineWidth', 1)
    xlabel('Time');
    ylabel('Normalized Signal');
    title('Top Correlated EIT Channels with Torque Peaks');
    legend('show');
    grid on;
    hold off;

end

% Function to compute correlation scores
function scores = compute_correlation_scores(eit_data, torque_data)
    num_channels = size(eit_data, 2);
    scores = zeros(1, num_channels);
    for i = 1:num_channels
        scores(i) = corr(eit_data(:, i), torque_data);
    end
end

% Function to compute correlation scores
function scores = compute_abs_correlation_scores(eit_data, torque_data)
    num_channels = size(eit_data, 2);
    scores = zeros(1, num_channels);
    for i = 1:num_channels
        scores(i) = abs(corr(eit_data(:, i), torque_data));
    end
end

% Function to compute signal-to-noise ratio (SNR) scores
function scores = compute_snr_scores(data)
    num_channels = size(data, 2);
    scores = zeros(1, num_channels);
    for i = 1:num_channels
        signal_power = rms(data(:, i))^2;
        noise_power = var(data(:, i) - mean(data(:, i)));
        scores(i) = 10 * log10(signal_power / noise_power);
    end
end

% Function to compute root-mean-square (RMS) scores
function scores = compute_rms_scores(data)
    num_channels = size(data, 2);
    scores = zeros(1, num_channels);
    for i = 1:num_channels
        scores(i) = rms(data(:, i));
    end
end

% Function to compute FFT similarity scores
function scores = compute_fft_similarity(data, load_data)
    num_channels = size(data, 2);
    scores = zeros(1, num_channels);

    % Compute FFT for load data
    load_fft = abs(fft(load_data));
    load_fft = load_fft(1:floor(length(load_fft)/2)); % Retain positive frequencies

    for i = 1:num_channels
        channel_fft = abs(fft(data(:, i)));
        channel_fft = channel_fft(1:floor(length(channel_fft)/2)); % Positive frequencies

        % Compute cosine similarity between FFTs
        scores(i) = dot(load_fft, channel_fft) / ...
                    (norm(load_fft) * norm(channel_fft));
    end
end

% Function to compute Dynamic Time Warping (DTW) scores
function scores = compute_dtw_scores(data, load_data)
    num_channels = size(data, 2);
    scores = zeros(1, num_channels);
    for i = 1:num_channels
        scores(i) = -dtw(data(:, i), load_data); % Negative as DTW gives a distance (lower is better)
    end
end