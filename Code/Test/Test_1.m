centre = 1.8;
sigma = 0.2;
scale = 0.1;
k=10;


select_fcn = @(x,y,z) exp(-(y - centre).^2 / (2 * sigma^2)) .*(scale * cos(k * (y-centre)));
plot(select_fcn(0,linspace(0,3.6,100),0))
