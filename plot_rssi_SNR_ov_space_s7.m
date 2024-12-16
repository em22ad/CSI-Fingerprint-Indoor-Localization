clear;
close all;

csi2 = zeros(1,3,30);
TX_Antenna=2;
RX_Antenna=3; %1:3
%figure;
%hold on;
rssi=zeros(13,8,3)+255;
rssi_rgb=zeros(13,8,3)+255;
avg_over=100;
fp=zeros(TX_Antenna*6,RX_Antenna*5);
fp_rssi=zeros(6,RX_Antenna*5);

for x=-6:6
    for y=-1:6
        try
            csi_trace = read_bf_file(sprintf('sample_data/csi_s7_%d_%d.dat',x,y));
            for i=1:avg_over %int16(size(csi_trace,1))
                csi_entry = csi_trace{i};
                if (size(csi_entry,2)>0) && (csi_entry.Nrx   == RX_Antenna) && (csi_entry.Ntx   == TX_Antenna)
                    sprintf('%1.1f',csi_entry.timestamp_low/67978);    
                    csi = get_scaled_csi(csi_entry(1,:,:));
                    for rx=1:RX_Antenna
                        xsh=(rx-1);
                        l=1;
                        for j=1:6
                            for k=1:5
                                if (rx == 1)
                                    fp_rssi(j,k+(5*xsh))=fp_rssi(j,k+(5*xsh))+csi_entry.rssi_a;
                                elseif (rx == 2)
                                    fp_rssi(j,k+(5*xsh))=fp_rssi(j,k+(5*xsh))+csi_entry.rssi_b;
                                else
                                    fp_rssi(j,k+(5*xsh))=fp_rssi(j,k+(5*xsh))+csi_entry.rssi_c;
                                end
                            end
                        end
                    end
                    %plot(db(abs(squeeze(csi2).')))
                end
            end
            
            fp_rssi=fp_rssi./avg_over;
            rssi(x+7,y+2,1)=csi_entry.rssi_a;
            rssi(x+7,y+2,2)=csi_entry.rssi_b;
            rssi(x+7,y+2,3)=csi_entry.rssi_c;
        catch exception
            disp('Error opening file');
            continue;
        end
    end
end
rssi=flipud(rssi);
rssi(isnan(rssi))=0;
for ch=1:3
    clear X;
    M=rssi(:,:,ch);
    [X(:,1),X(:,2)] = find(M~=255);
    V=zeros(size(X,1),1);
    for i=1:size(X(:,1),1)
        V(i)=M(X(i,1),X(i,2));
    end
    F = scatteredInterpolant(X,V);
    F.Method = 'natural';
    O=zeros(size(M));
    for i=1:size(M,1)
        for j=1:size(M,2)
            O(i,j)=F([i j]);
        end
    end
    rssi(:,:,ch)=O;
end

for ch=1:3
    max_rgb=max(max(rssi(:,:,ch)));
    min_rgb=min(min(rssi(:,:,ch)));
    for x=-6:6
        for y=-1:6
            rssi_rgb(x+7,y+2,ch)=interp1([min_rgb,max_rgb],[0,255],rssi(x+7,y+2,ch));
            %rssi_rgb./max(rssi_rgb(:));
        end
    end
end
% Extract the individual red, green, and blue color channels.
redChannel = rssi_rgb(:, :, 1);
greenChannel = rssi_rgb(:, :, 2);
blueChannel = rssi_rgb(:, :, 3);
windowSize = 1;
kernel = ones(windowSize) / windowSize ^ 2;
% Blur the individual color channels.
smoothedPictureR = conv2(double(redChannel), kernel, 'same');
smoothedPictureG = conv2(double(greenChannel), kernel, 'same');
smoothedPictureB = conv2(double(blueChannel), kernel, 'same');
% Recombine separate color channels into a single, true color RGB image.
smoothrgbImage = uint8(cat(3, smoothedPictureR , smoothedPictureG, smoothedPictureB));
J = imresize(smoothrgbImage, 10.0);
imshow(J);
%imshow(rssi_rgb);
%I_rssi = mat2gray(fp_rssi);
figure
[X,Y] = meshgrid(1:13,1:8);
Z1 = rssi(:,:,1)';
Z2 = rssi(:,:,2)';
Z3 = rssi(:,:,3)';
Z=(Z1+Z2+Z3)/3.0;
C = X.*Y;
surf(X,Y,Z)
colorbar
title('Average RSSI strength over Space 13m x 8m','FontSize',24)
xlabel('X-axis in meters') 
ylabel('Y-axis in meters')
zlabel('RSSI strength')
%imshow(fp_rssi.*2, [0 150]);