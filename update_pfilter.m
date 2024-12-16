function w=update_pfilter(obs,p)

var1=(1-obs(1,1))*1.0;
%var1=0.5;
if (var1 <= 0)
    var1=0.001;
end

h=(1-obs(6,1))*1.2;
if (h <= 0)
    h=0.001;
end

for s=1:size(p,2)
    sum_expL=((p(s).xL-obs(3,1))^2+(p(s).yL-obs(4,1))^2)/(2*var1^2);
    sum_expR=((p(s).xR-obs(3,1))^2+(p(s).yR-obs(4,1))^2)/(2*var1^2);
    
    pL=(1/(sqrt(2*pi)*var1))*exp(-sum_expL);
    pR=(1/(sqrt(2*pi)*var1))*exp(-sum_expR);
    
    d0=(p(s).xR-p(s).xL)*cos(obs(5,1))+(p(s).yR-p(s).yL)*sin(obs(5,1));
    %d0=(p(s).xR-p(s).xL)*cos(p(s).theta)+(p(s).yR-p(s).yL)*sin(p(s).theta); h=0.8;
    r0=-p(s).Stride*cos(p(s).ph);
    
    pB=(1/(sqrt(2*pi)*h))*exp(-((d0-r0)^2)/(2*h^2));
    if (pR < 0) || isnan(pR)
        pR
    end
    
    if (pL < 0) || isnan(pL)
        pL
    end
    if (pB < 0) || isnan(pB)
        pB
    end

    w(s,1)=pR*pL*pB;
end

total=sum(w(:,1));
if (total < 0.0001)
    w=w./0.0001;
else
    w=w./total;    
end


