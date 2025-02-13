centre = 1.5;
sigma = 0.3;
k=10;
y= linspace(0,3,100);
select_fcn = exp(-(y - centre).^2 / (2 * sigma^2)) .*(cos(k * (y-centre)));
hold on;
plot(select_fcn)
