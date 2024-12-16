function plot_heatmap(data2d)
%// Define integer grid of coordinates for the above data
[X,Y] = meshgrid(1:size(data2d,2), 1:size(data2d,1));

%// Define a finer grid of points
[X2,Y2] = meshgrid(1:0.01:size(data2d,2), 1:0.01:size(data2d,1));

%// Interpolate the data and show the output
outData = interp2(X, Y, data2d, X2, Y2, 'linear');
imagesc(outData);

%// Cosmetic changes for the axes
set(gca, 'XTick', linspace(1,size(X2,2),size(X,2))); 
set(gca, 'YTick', linspace(1,size(X2,1),size(X,1)));
set(gca, 'XTickLabel', 1:size(X,2));
set(gca, 'YTickLabel', 1:size(X,1));

%// Add colour bar
colorbar;