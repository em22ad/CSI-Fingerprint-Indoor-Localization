data=squeeze(fp_csi(1,4,1,:));
sz=size(data,1);
x = mean(data);
x2 = median(data);

subset=floor(sz/10);
for i=1:subset
    x3(i) = median(data(randsample(sz,floor(subset/3))'));
    
end
x4 = mean(x3);

std_up=1.0*std(data);
hold on;
plot(repmat(x+std_up,1,length(data)),'-g');
plot(repmat(x-std_up,1,length(data)),'-g');
plot(repmat(x,1,length(data)),'-r');
plot(repmat(x2,1,length(data)),'-k');
plot(repmat(x4,1,length(data)),'-m');
plot(data,'-b');
hold off;