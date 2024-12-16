clear all;
rng('default')  % For reproducibility
y = [2.8284 0 0 2.4495 0 0 2.6458 2.6458 0 3 0 0 0 2 0 0 0 0 1.4142 0 1.4142 0 0 0 0 1 0 0 0 0 1.4142 1 0 0 0 2.2361 3 0 2.2361 0];
cdfplot(y)
hold on
y1 = [2.8284 2.82842 2.8284 0 2.4495 2.4495 2.6458 2.6458 0 4 0 2.2361 0 0 0 0 2 0 2.2361 0 3 0 0 3 0 2.8284 0 0 0 0 2 2.8284 0 0 0 3 2 0 2 0];
% x = linspace(1:1:31);
% plot(x,y1)
cdfplot(y1)
legend('distance between transition vector centers','distance between grid cell centers','Location','best')
xlabel('Positioning Error [m]');
ylabel('Test Dataset Percentage [%]');
hold off