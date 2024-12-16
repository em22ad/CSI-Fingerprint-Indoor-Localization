clear;
drawArrow = @(x,y,r,g,b) quiver( x(1),y(1),x(2)-x(1),y(2)-y(1),0,'color',[r g b]);
load('walk_obs.mat');
hyp=1;
%TBD: normalize variance between 0 and 1
normVar = obs{hyp}(1,:) - min(obs{hyp}(1,:));
normVar = normVar ./ max(normVar(:));
normthVar = obs{hyp}(6,:) - min(obs{hyp}(6,:));
normthVar = normthVar ./ max(normthVar(:));

%initialize particle value
M=1000;
total_obs=size(obs{hyp},2);
%Noise params
linear_fact=1.0;
turn_fact=1.0;
foot_noise=0.4; %lets say there is a 0.2 m radius towards which this particle will be senstive for updates 

str_noise=linear_fact*0.20; %lets assume stride varies between 0.05 m to 0.45 m i.e. 0.40m
tm_stamp_noise=0.1; %The time between two time steps can be between 0.65 and 0.85 with normal distribution
time_period_noise=0.15; %Expect a 0.05 sec change in period between two steps %It takes avg 0.75 sec to take a step
ph_noise=linear_fact*0.25*2*pi;%Expect 45 degree phase change noise.
theta_noise=turn_fact*0.8; %Expect -90 to 90 degree change between two steps with normal distribution

sum_xL=0;
sum_xR=0;
sum_yL=0;
sum_yR=0;
for j=1:M
    r=rand();
    r2=rand();
    p(j).T=0.1+r*2.0; %randomize bw 0.1sec to 2.1sec 
    p(j).Stride=0.05+r*0.25; %randomize bw 0.05m to 0.30m
    p(j).theta=0.1+r2*1.6; %randomize bw 5 deg  to 180 deg
    p(j).xL=obs{hyp}(3,1)+(-5+r*10); %randomize bw -5 to 5 m around prior belief
    p(j).yL=obs{hyp}(4,1)+(-5+r2*10); %randomize bw -5 to 5 m around prior belief
    p(j).xR=-0.25+p(j).xL+r*0.5; %randomize bw -5 to 5 m around prior belief of left foot
    p(j).yR=-0.25+p(j).yL+r2*0.5; %randomize bw -5 to 5 m around prior belief of left foot
    p(j).ph=r*2*pi; %randomize bw 0 to 2pi
    p(j).r=rand();
    p(j).g=rand();
    p(j).b=rand();
    
    sum_xL=sum_xL + p(j).xL;
    sum_xR=sum_xR + p(j).xR;
    sum_yL=sum_yL + p(j).yL;
    sum_yR=sum_yR + p(j).yR;
end
peds=1;
w=randfixedsum(M,peds,1.0,0,1.0);
%randsample(1:M,M,false,w); %uses a vector of nonnegative weights, w, of the same length as the vector population, to determine the probability that a value population(i) is selected as an entry for y.

xhat_ser=[];
yhat_ser=[];
x_ser=[];
y_ser=[];

% sumw=sum(w(:,1));
% xhat=(sum_xL+sum_xR)/(M*(2.0*sumw));
% yhat=(sum_yL+sum_yR)/(M*(2.0*sumw)); 

figure;
hold on
for i=2:total_obs %we will only simulate until the observations are exhausted
    resamp=0;
    if (mod(i,3)== 0)
        single=[normVar(i);obs{hyp}(2:5,i);normthVar(i)];
        %single=obs{1,1}(:,i);
        w=update_pfilter(single,p);
        resamp=1;
    end
    
    sum_xL=0;
    sum_xR=0;
    sum_yL=0;
    sum_yR=0;
    
%     1   3  4 
%     2   4  4
%     *3  4  4
%     *4  4  3
%     *5  3  3
%     6   5  4
%     7   5  3

    for j=1:M
        if (resamp == 1)
            idx=randsample(1:M,1,true,w);
        else
            idx=j;
        end
        
        rnd1=randn/4.3;
        rnd2=randn/4.3;
        Striden=p(idx).Stride+(rnd2*str_noise);
        dt=0.75+(rnd2*tm_stamp_noise); %It takes avg 0.75 sec to take a step
        phn=p(idx).ph+2*pi*(dt/p(idx).T)+rnd2*ph_noise;
        md=p(idx).Stride*abs(cos(phn)-cos(p(idx).ph));
        if (mod(phn,2*pi) > pi)
            rL=md+(rnd1*foot_noise);rR=(rnd1*foot_noise);
        else
            rR=md+(rnd1*foot_noise);rL=(rnd1*foot_noise);
        end

        xLn=p(idx).xL+rL*cos(p(idx).theta)-rL*sin(p(idx).theta);
        yLn=p(idx).yL+rL*sin(p(idx).theta)+rL*cos(p(idx).theta);
        xRn=p(idx).xR+rR*cos(p(idx).theta)-rR*sin(p(idx).theta);
        yRn=p(idx).yR+rR*sin(p(idx).theta)+rR*cos(p(idx).theta);

        p(j).T=p(idx).T+(rnd2*time_period_noise);
        if (p(j).T < 0.1) %Do not let get time period go in negative
            p(j).T=0.1;
        end
 
        if (p(j).T > 5.0) %Time period can not be longer than a value
            p(j).T=5.0;
        end
        p(j).theta=p(idx).theta+((randn/4.3)*theta_noise); 

%         x1 = [p(j).xL xLn];
%         y1 = [p(j).yL yLn];
%         plot(x1,y1,'.','color',[192/255 192/255 192/255])%[p(j).r p(j).g p(j).b]);
%         x2 = [p(j).xR xRn];
%         y2 = [p(j).yR yRn];
%         plot(x2,y2,'.','color',[128/255 128/255 128/255])%[p(j).r p(j).g p(j).b]);
        %xlim([-5, 5])
        %ylim([-5, 5])

        p(j).Stride=Striden;
        p(j).xL=xLn;
        p(j).yL=yLn;   
        p(j).xR=xRn;
        p(j).yR=yRn;  
        p(j).ph=phn;
        
        sum_xL=sum_xL + p(j).xL;
        sum_xR=sum_xR + p(j).xR;
        
        sum_yL=sum_yL + p(j).yL;
        sum_yR=sum_yR + p(j).yR;
    end
    
    %sumw=sum(w(:,1));    
    xhat_n=(sum_xL+sum_xR)/(2*M);
    yhat_n=(sum_yL+sum_yR)/(2*M);  

    %drawArrow(x,y,1,0,0);
    xhat_ser=[xhat_ser;xhat_n];
    yhat_ser=[yhat_ser;yhat_n];
    
    %plot(xt,yt,'d-b');
    x_ser=[x_ser;obs{hyp}(3,i)];
    y_ser=[y_ser;obs{hyp}(4,i)];
    
    xhat=xhat_n; 
    yhat=yhat_n;
    
%     hold off;
%     bt=waitforbuttonpress;
%     bt_val=double(get(gcf,'CurrentCharacter'));
%     if (bt_val == 32)
%        break;
%     end
end

%ys=yhat_ser;
%xs=xhat_ser;
ys=smooth(yhat_ser(:,1));
xs=smooth(xhat_ser(:,1));
ys=smooth(ys);
xs=smooth(xs);
plot(xs,ys,'-r','LineWidth',3);

ys=smooth(y_ser);
xs=smooth(x_ser);
ys=smooth(ys);
xs=smooth(xs);
plot(xs,ys,'-b','LineWidth',3);%'markersize',5);
axis equal
hold off;