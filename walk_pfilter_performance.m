clear;
load('walk_est.mat');
% for hyp=1:size(obs,1)%size(map,2)
%     xhat_ser{hyp}(:)=smooth(xhat_ser{hyp}(:));
%     yhat_ser{hyp}(:)=smooth(yhat_ser{hyp}(:));
%     xhat_ser{hyp}(:)=smooth(xhat_ser{hyp}(:));
%     yhat_ser{hyp}(:)=smooth(yhat_ser{hyp}(:));
% end
hyp=1;
t = linspace(0,size(xhat_ser{hyp}(:),1),size(xhat_ser{hyp}(:),1));
subplot(4,1,1)
y = obs{hyp,1}(3,:);
plot(t,y,':b','LineWidth',3);
hold on
y2= xhat_ser{hyp}(:);
plot(t,y2,'-r');

subplot(4,1,2)
y = obs{hyp,1}(4,:);
plot(t,y,':b','LineWidth',3);
hold on
y2= yhat_ser{hyp}(:);
plot(t,y2,'-r');

subplot(4,1,3)
y = rad2deg(obs{hyp,1}(5,:));
plot(t,y,':b','LineWidth',3);
hold on
y2= rad2deg(thetahat_ser{hyp}(:))-290;
plot(t,y2,'-r');

subplot(4,1,4)
y = obs{hyp,1}(7,:);
plot(t,y,':b','LineWidth',3);
hold on
y2= stridehat_ser{hyp}(:);
plot(t,y2,'-r');
