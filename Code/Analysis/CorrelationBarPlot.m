figure();
corNames = ["Step","Linear","Gaussian Differential", "Modulated Gaussian"]; 
corScores = [0.23516,0.22388,0.24002,0.34881];

bar(corNames,corScores, 'black')
fontsize(12,"points")
title( "Model to Ground Truth Cross Correlation Scores",FontSize=20)
xlabel('Model',FontSize=16)
ylabel('Cross Correlation',FontSize=16)
ylim([0,1]);