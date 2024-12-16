clear;
drawArrow = @(x,y,r,g,b) quiver( x(1),y(1),x(2)-x(1),y(2)-y(1),0,'color',[r g b]);
load('walk_obs.mat');
%TBD: normalize variance between 0 and 1
hyp_no=size(obs,1);
all_var=[];
all_thvar=[];
for hyp=1:hyp_no
    all_var=[all_var obs{hyp}(1,:)];
    all_thvar=[all_thvar obs{hyp}(6,:)];
end

for hyp=1:hyp_no
    normVar{hyp} = obs{hyp}(1,:) - min(all_var);
    normVar{hyp} = normVar{hyp} ./ max(all_var);
    normthVar{hyp} = obs{hyp}(6,:) - min(all_thvar);
    normthVar{hyp} = normthVar{hyp} ./ max(all_thvar);
end
%initialize particle value
M=1000;
USE_IMU_TH=0;
total_obs=size(obs{1},2);%assume all hyp have same obs
%Noise params
linear_fact=1.0;
turn_fact=1.5;
foot_noise=0.2; %lets say there is a 0.2 m radius towards which this particle will be senstive for updates 

str_noise=linear_fact*0.20; %lets assume stride varies between 0.05 m to 0.45 m i.e. 0.40m
tm_stamp_noise=0.1; %The time between two time steps can be between 0.65 and 0.85 with normal distribution
time_period_noise=0.15; %Expect a 0.05 sec change in period between two steps %It takes avg 0.75 sec to take a step
ph_noise=linear_fact*0.25*2*pi;%Expect 140 degree phase change noise.
theta_noise=turn_fact*0.8; %Expect -90 to 90 degree change between two steps with normal distribution

imu_th=0;
sum_xL=0;
sum_xR=0;
sum_yL=0;
sum_yR=0;

for hyp=1:hyp_no
    for j=1:M
        r=rand();
        r2=rand();
        p(hyp,j).T=0.1+r*2.0; %randomize bw 0.1sec to 2.1sec 
        p(hyp,j).Stride=0.05+r*0.25; %randomize bw 0.05m to 0.30m
        p(hyp,j).theta=0.1+r2*1.6; %randomize bw 5 deg  to 180 deg
        p(hyp,j).xL=obs{hyp}(3,1)+(-0.25+r*0.5); %randomize bw -1 to 1 m around prior belief
        p(hyp,j).yL=obs{hyp}(4,1)+(-0.25+r2*0.5); %randomize bw -1 to 1 m around prior belief
        p(hyp,j).xR=-0.25+p(hyp,j).xL+r*0.5; %randomize bw -1 to 1 m around prior belief of left foot
        p(hyp,j).yR=-0.25+p(hyp,j).yL+r2*0.5; %randomize bw -1to 1 m around prior belief of left foot
        p(hyp,j).ph=r*2*pi; %randomize bw 0 to 2pi
        p(hyp,j).r=rand();
        p(hyp,j).g=rand();
        p(hyp,j).b=rand();

%         sum_xL=sum_xL + p(hyp,j).xL;
%         sum_xR=sum_xR + p(hyp,j).xR;
%         sum_yL=sum_yL + p(hyp,j).yL;
%         sum_yR=sum_yR + p(hyp,j).yR;
    end
end

peds=hyp_no;
w=randfixedsum(M,peds,1.0,0,1.0);
%randsample(1:M,M,false,w); %uses a vector of nonnegative weights, w, of the same length as the vector population, to determine the probability that a value population(i) is selected as an entry for y.

for hyp=1:hyp_no
    xhat_ser{hyp}=[];
    yhat_ser{hyp}=[];
    thetahat_ser{hyp}=[];
    stridehat_ser{hyp}=[];
    
    x_ser{hyp}=[];
    y_ser{hyp}=[];
    rgb{hyp}.r=rand();%0.2*hyp;
    rgb{hyp}.g=rand();%0.2*hyp;
    rgb{hyp}.b=rand();%1.0;
    rgb2{hyp}.r=rgb{hyp}.r;%0.2*hyp;
    rgb2{hyp}.g=rgb{hyp}.g;%0.2*hyp;
    rgb2{hyp}.b=rgb{hyp}.b;%1.0;
    lw{hyp}=3;
end

figure;
hold on
map=1:hyp_no;
for i=1:total_obs %we will only simulate until the observations are exhausted
    resamp=0;
    if (mod(i,3)== 0) && (size(map,2) > 0)
        hyp_cols=[]; %6x4 matrix
        for hyp=1:size(map,2)
            col=[normVar{map(hyp)}(i);obs{map(hyp)}(2:5,i);normthVar{map(hyp)}(i)];
            hyp_cols=[hyp_cols col];
        end
        w=update_pfilter_multihyp_tuned4cnn(hyp_cols,p,map,USE_IMU_TH);
        resamp=1;
    end
    
    for hyp=1:size(map,2)
        sum_xL=0;
        sum_xR=0;
        sum_yL=0;
        sum_yR=0;
        sum_theta=0;
        sum_stride=0;
        
        sumw=sum(w(:,map(hyp)));
        if (sumw <= 0)
            rgb{map(hyp)}.r=0.8;
            rgb{map(hyp)}.g=0.8;
            rgb{map(hyp)}.b=0.8;
            lw{map(hyp)}=1;
            idx=find(map~=map(hyp));
            map=map(idx);
            break;
        end

        for j=1:M
            if (resamp == 1)
                idx=randsample(1:M,1,true,w(:,map(hyp)));
            else
                idx=j;
            end

            rnd1=randn/4.3;
            rnd2=randn/4.3;
            Striden=p(map(hyp),idx).Stride+(rnd2*str_noise);
            dt=0.75+(rnd2*tm_stamp_noise); %It takes avg 0.75 sec to take a step
            ph_n=p(map(hyp),idx).ph+2*pi*(dt/p(map(hyp),idx).T)+rnd2*ph_noise;
            md=p(map(hyp),idx).Stride*abs(cos(ph_n)-cos(p(map(hyp),idx).ph));
            if (mod(ph_n,2*pi) > pi)
                rL=md+(rnd1*foot_noise);rR=(rnd1*foot_noise);
            else
                rR=md+(rnd1*foot_noise);rL=(rnd1*foot_noise);
            end

            xLn=p(map(hyp),idx).xL+rL*cos(p(map(hyp),idx).theta)-rL*sin(p(map(hyp),idx).theta);
            yLn=p(map(hyp),idx).yL+rL*sin(p(map(hyp),idx).theta)+rL*cos(p(map(hyp),idx).theta);
            xRn=p(map(hyp),idx).xR+rR*cos(p(map(hyp),idx).theta)-rR*sin(p(map(hyp),idx).theta);
            yRn=p(map(hyp),idx).yR+rR*sin(p(map(hyp),idx).theta)+rR*cos(p(map(hyp),idx).theta);

            p(map(hyp),j).T=p(map(hyp),idx).T+(rnd2*time_period_noise);
            p(map(hyp),j).theta=p(map(hyp),idx).theta+((randn/4.3)*theta_noise); 

            p(map(hyp),j).Stride=Striden;
            p(map(hyp),j).xL=xLn;
            p(map(hyp),j).yL=yLn;   
            p(map(hyp),j).xR=xRn;
            p(map(hyp),j).yR=yRn;  
            p(map(hyp),j).ph=ph_n;

            sum_xL=sum_xL + p(map(hyp),j).xL;
            sum_xR=sum_xR + p(map(hyp),j).xR;        
            sum_yL=sum_yL + p(map(hyp),j).yL;
            sum_yR=sum_yR + p(map(hyp),j).yR;
            sum_theta=sum_theta + p(map(hyp),j).theta;
            sum_stride=sum_stride + p(map(hyp),j).Stride;
        end
        %sumw=sum(w(:,map(hyp)));    
        xhat_n=(sum_xL+sum_xR)/(2*M);
        yhat_n=(sum_yL+sum_yR)/(2*M);
        thetahat_n=sum_theta/M;
        stridehat_n=sum_stride/M;

        xhat_ser{map(hyp)}=[xhat_ser{map(hyp)};xhat_n];
        yhat_ser{map(hyp)}=[yhat_ser{map(hyp)};yhat_n];
        thetahat_ser{map(hyp)}=[thetahat_ser{map(hyp)};thetahat_n];
        stridehat_ser{map(hyp)}=[stridehat_ser{map(hyp)};stridehat_n];
    end    
    
%     hold off;
%     bt=waitforbuttonpress;
%     bt_val=double(get(gcf,'CurrentCharacter'));
%     if (bt_val == 32)
%        break;
%     end
end
%%
for hyp=1:size(obs,1)%size(map,2)
    %ys=yhat_ser;
    %xs=xhat_ser;
    xhat_ser{hyp}(:)=smooth(xhat_ser{hyp}(:));
    yhat_ser{hyp}(:)=smooth(yhat_ser{hyp}(:));
    xhat_ser{hyp}(:)=smooth(xhat_ser{hyp}(:));
    yhat_ser{hyp}(:)=smooth(yhat_ser{hyp}(:));
    
    plot(xhat_ser{hyp}(:),yhat_ser{hyp}(:),'color',[rgb{hyp}.r rgb{hyp}.g rgb{hyp}.b],'LineWidth',lw{hyp});

    obs1{hyp,1}(3,:)=obs{hyp,1}(3,:);
    obs1{hyp,1}(4,:)=obs{hyp,1}(4,:);    
    
    plot(obs1{hyp,1}(3,:),obs1{hyp,1}(4,:),':','color',[rgb2{hyp}.r rgb2{hyp}.g rgb2{hyp}.b],'LineWidth',3);%'markersize',5);    

    thetahat_ser{hyp}(:)=smooth(thetahat_ser{hyp}(:));
    thetahat_ser{hyp}(:)=smooth(thetahat_ser{hyp}(:));
    stridehat_ser{hyp}(:)=smooth(stridehat_ser{hyp}(:));
    stridehat_ser{hyp}(:)=smooth(stridehat_ser{hyp}(:));
end
axis equal
hold off;
save('walk_est.mat','obs','xhat_ser','yhat_ser','thetahat_ser','stridehat_ser');