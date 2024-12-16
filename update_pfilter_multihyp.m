function w=update_pfilter_multihyp(hyp_cols,p,map,USE_IMU_TH)
%hyp_cols 6x4 matrix
hyp_no=size(hyp_cols,2);
%var1=zeros(1,100);
%h=zeros(1,100);
for hyp=1:size(map,2)
    %If you make var1 high, model gets a lot of flexibility to accept many
    %invalid poses as valid
    var1(map(hyp))=(1-hyp_cols(1,hyp))*1.5;
    
    %if you make h very low, model will have very few choices to follow a
    %model and may not complete tracking. Set it very high and model is so 
    %libral that it takes twirls while moving forward
    %to take twirls 
    if (USE_IMU_TH == 1)
        h(map(hyp))=(1-hyp_cols(6,hyp))*0.3;
    else
        h(map(hyp))=0.3;
    end
end

for s=1:size(p,2)    
    for hyp=1:size(map,2)
        termL=(((p(map(hyp),s).xL-hyp_cols(3,hyp))^2+(p(map(hyp),s).yL-hyp_cols(4,hyp))^2)/(2*var1(map(hyp))^2));
        pL=(1/(sqrt(2*pi)*var1(map(hyp))))*exp(-termL);
        termR=(((p(map(hyp),s).xR-hyp_cols(3,hyp))^2+(p(map(hyp),s).yR-hyp_cols(4,hyp))^2)/(2*var1(map(hyp))^2));
        pR=(1/(sqrt(2*pi)*var1(map(hyp))))*exp(-termR);
        r0=-p(map(hyp),s).Stride*cos(p(map(hyp),s).ph);
        
        if (USE_IMU_TH == 1)
            d0=(p(map(hyp),s).xR-p(map(hyp),s).xL)*cos(hyp_cols(5,hyp))+(p(map(hyp),s).yR-p(map(hyp),s).yL)*sin(hyp_cols(5,hyp));
        else
            d0=(p(map(hyp),s).xR-p(map(hyp),s).xL)*cos(p(map(hyp),s).theta)+(p(map(hyp),s).yR-p(map(hyp),s).yL)*sin(p(map(hyp),s).theta);
        end
        
        pB=(1/(sqrt(2*pi)*h(map(hyp))))*exp(-((d0-r0)^2))/(2*h(map(hyp))^2);
        if (pR < 0) || isnan(pR)
            pR
        end
        if (pL < 0) || isnan(pL)
            pL
        end
        if (pB < 0) || isnan(pB)
            pB
        end
        w(s,map(hyp))=pR*pL*pB;
    end
end

for hyp=1:size(map,2)
    total=sum(w(:,map(hyp)));
    if (total < 0.0000001)
        w(:,map(hyp))=w(:,map(hyp))./0.0000001;
    else
        w(:,map(hyp))=w(:,map(hyp))./total;    
    end
end

%[f, x] = hist(w,size(p,2)); % Create histogram from a normal distribution.
% g = 1 / sqrt(2 * pi) * exp(-0.5 * x .^ 2); % pdf of the normal distribution
% 
% % METHOD 1: DIVIDE BY SUM
% figure(1)
% bar(x, f / sum(f)); hold on
% plot(x, g, 'r'); hold off
% 
% % METHOD 2: DIVIDE BY AREA
% figure(2)
% bar(x, f / trapz(x, f)); hold on
% plot(x, g, 'r'); hold off

%w=f / sum(f);
%w=f / trapz(x, f);