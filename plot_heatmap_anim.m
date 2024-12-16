function plot_heatmap_anim(data)
%// Define integer grid of coordinates for the above data
[X,Y] = meshgrid(1:size(data,2), 1:size(data,1));

%// Define a finer grid of points
[X2,Y2] = meshgrid(1:0.01:size(data,2), 1:0.01:size(data,1));


figure
%// Cosmetic changes for the axes
set(gca, 'XTick', linspace(1,size(X2,2),size(X,2))); 
set(gca, 'YTick', linspace(1,size(X2,1),size(X,1)));
set(gca, 'XTickLabel', 1:size(X,2));
set(gca, 'YTickLabel', 1:size(X,1));

%// Add colour bar
colorbar;
for ch=1:size(data,3)
    %// Interpolate the data and show the output
    outData = interp2(X, Y, data(:,:,ch), X2, Y2, 'linear');
    imagesc(outData);
    hold off;
    pause(1);
end

