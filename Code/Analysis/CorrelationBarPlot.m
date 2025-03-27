figure();
corNames = ["Step","Linear","Gaussian Differential", "Modulated Gaussian","Linear with a Cutoff"]; 
% corScores = [0.23516,0.22388,0.24002,0.34881]; %OLD SCORES 
% corScores = [0.603556147520199,0.656713102600582,0.661521712438642,0.537041585145483]; %Scores allowing best shift to be found
% corScores = [0.1240,0.1934,0.0077,0.3547]; %NEW SCORES using envelope correlation
corScores = [0.65126,0.60932,0.70668,0.58045,0.6258]; %EVEN NEWER
mseScores = [0.048429,0.082369,0.039878,0.062376,0.05579]; %matching MSE Scores
bar(corNames,corScores, 'black')
fontsize(12,"points")
title( "Model to Ground Truth Cross Correlation Scores",FontSize=20)
xlabel('Model',FontSize=16)
ylabel('Cross Correlation',FontSize=16)
ylim([0,1]);