clear;
Left_Data = load("40mm_right.mat").plotthis_right;
Load_Data = smoothdata(abs(load('40mm_truth.mat').plotthis_truth));

plot(Load_Data(:,1),Load_Data(:,2))

EIT = Left_Data(:,2:end);
EIT = EIT(:,~all(EIT==0));
EIT_Time = Left_Data(:,1);
[pks,locs] = findpeaks(Load_Data(:,2),'MinPeakHeight',0.005);
torqueTimes = Load_Data(locs,1);

closest_values = interp1(EIT_Time, 1:length(EIT_Time), torqueTimes, 'next', 'extrap');
    
% Convert NaN values (out-of-range) to valid indices
closest_values(isnan(closest_values)) = length(EIT_Time);

% Convert to integer indices
closest_values = ceil(closest_values);

figure();hold on;
plot(abs(EIT(closest_values(1),:) - EIT(1,:))) ;
hold off;
hom = EIT(1,:);
trainTorquePeaks = pks(1:2);
testTorquePeaks = pks(3:length(pks));
save("Code\\SavedVariables\\TorqueFitting\Torque_40mm.mat","trainTorquePeaks","testTorquePeaks")
for i = 1:2
    data = EIT(closest_values(i),:);
    % plot(data)
    data_diff = abs(data-hom);
    % plot(data_diff)
    save(sprintf("Code\\SavedVariables\\TorqueFitting_40mm\\Train_%i",i),"data_diff");
    disp(i);
end
for i=3:length(pks)
    data = EIT(closest_values(i),:);
    data_diff = data-hom;
    save(sprintf("Code\\SavedVariables\\TorqueFitting_40mm\\Test_%i",i),"data_diff");
    disp(i);
end 



