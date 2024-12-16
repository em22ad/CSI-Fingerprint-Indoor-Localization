drawArrow = @(x,y,r,g,b) quiver( x(1),y(1),x(2)-x(1),y(2)-y(1),0,'color',[r g b]);
%save particle trajectories into a matrix
obs={};

%initialize particle value
particle_card=4;
steps=600;

%Noise params
noise_fact=4.0;
for j=1:particle_card
    foot_noise(j)=0.75*(1+(j-1)*noise_fact);
    str_noise(j)=0.15*(1+(j-1)*noise_fact); %lets assume stride varies between 0.30 m to 0.45 m i.e. 0.15m
    tm_stamp_noise(j)=0.1*(1+(j-1)*noise_fact); %The time between two time steps can be between 0.8 and 1.0 with normal distribution
    time_period_noise(j)=0.15*(1+(j-1)*noise_fact); %Expect a 0.3 sec change in period between two steps %It takes avg 0.75 sec to take a step
    ph_noise(j)=0.05*2*pi*(1+(j-1)*noise_fact); %Expect 35 degree phase change noise. 
    theta_noise(j)=0.4*(1+(j-1)*noise_fact); %Expect -22 to 22 degree change between two steps with normal distribution
end

r=rand();
r2=rand();
for j=1:particle_card
%  r=rand();
%  r2=rand();
    p(j).T=0.25+r*0.75; %randomize bw 0.25sec to 1.0sec 
    p(j).Stride=0.20+r*0.25; %randomize bw 0.2m to 0.45m
    p(j).theta=0.1+r2*1.6; %randomize bw 5 deg  to 180 deg
    p(j).x=-50+r*100; %randomize bw -50 to 50 m
    p(j).y=-50+r2*100; %randomize bw -50 to 50 m
    p(j).ph=r*2*pi; %randomize bw 0 to 2pi
    p(j).speed=0;
end

figure;

for j=1:particle_card
    r=rand();
    g=rand();
    b=rand();
    part_traj=[];
    t=0.0;
    for i=1:steps
        rnd2=(randn/4.3);
        Striden=p(j).Stride+(rnd2*str_noise(j));
        rnd4=(randn/4.3);
        tn=t+0.75+(rnd4*tm_stamp_noise(j)); %It takes avg 0.75 sec to take a step
        dt=tn-t;
        ph_n=p(j).ph+2*pi*(dt/p(j).T)+(rnd2*ph_noise(j));
        rnd5=(randn/4.3);
        md=p(j).Stride*abs(cos(ph_n)-cos(p(j).ph))+(rnd5*foot_noise(j));
        xn=p(j).x+md*cos(p(j).theta);
        yn=p(j).y+md*sin(p(j).theta);

        p(j).T=p(j).T+(rnd2*time_period_noise(j));

        if (p(j).T < 0.4) %Do not let get time period go in negative
            p(j).T=0.3;
        end


        if (p(j).T > 3.0) %Time period can not be longer than a value
            p(j).T=1.1;
        end
        
        rnd3=(randn/4.3);
        p(j).theta=p(j).theta+(rnd3*theta_noise(j)); 

        x1 = [p(j).x xn];
        y1 = [p(j).y yn];
        drawArrow(x1,y1,r,g,b); hold on
        vari=(abs(rnd3*theta_noise(j))+abs(rnd4*tm_stamp_noise(j))+abs(rnd2*time_period_noise(j))+abs(rnd5*foot_noise(j))+abs(rnd2*ph_noise(j))+abs(rnd2*str_noise(j)));%/(theta_noise(j)+tm_stamp_noise(j)+time_period_noise(j)+foot_noise(j)+ph_noise(j)+str_noise(j));
        part_traj=[part_traj [vari;tn;xn;yn;p(j).theta; abs(rnd3*theta_noise(j)); Striden]];
        %xlim([-5, 5])
        %ylim([-5, 5])
        
        p(j).ph=ph_n;
        p(j).Stride=Striden;
        p(j).speed=p(j).speed+(p(j).Stride/dt);
        p(j).x=xn; 
        p(j).y=yn;   
        t=tn;
    end
    obs{j,1}=part_traj;
end
axis equal
hold off;

for j=1:particle_card
    p(j).speed=p(j).speed/steps;
end
save('walk_obs.mat','obs');