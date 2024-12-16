clear;
close all;
FigPathName='C:/Users/em22a/OneDrive/Desktop/comparison/phaseovspace/';
%DSPathName='C:/Users/em22a/Downloads/coarse_localization_esp/csi_preproc_16freq/';
DSPathName='C:/Users/em22a/Downloads/coarse_localization_esp/csi_2dch/';
PathName='../CSI_dataset/'; ds_tag='0';
%PathName='../CSI_dataset_test/';ds_tag='3';
%PathName='../CSI_dataset_test/03_29_19_1_10pm/'; ds_tag='1';
%PathName='../CSI_dataset_test/03_29_19_8_35pm/'; ds_tag='2';
%PathName='../CSI_dataset_test/04_13_19_7_15pm/'; ds_tag='4';
FREQ=[1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30];
%FREQ=[4 5 10 14 15 20 21 22 23 24 25 26 27 28 29 30];
TX=[1 2 3];
RX=[1 2 3];
Ntx=size(TX,2);
Nrx=size(RX,2); %1:3

csi2 = zeros(1,3,30);
fp_csi=zeros(13,13,Ntx*Nrx*30)+255;

st=0.0;%start at 0%th observation
en=1.0;%1.0 mean 100% observations

for x=-6:2:6
    for y=-6:2:6
        try
            FileName{1}=sprintf('csi_ng_%d_%d.dat',x,y);
            if ~exist(strcat(PathName,char(FileName(1))), 'file')
                errorMessage=sprintf('File: %s not found.',strcat(PathName,char(FileName(1))));
                disp(errorMessage);
                continue;
            end
            
            csi_trace = read_bf_file(strcat(PathName,char(FileName(1))));
            csi2 = zeros(1,3,30);    
            for i=1+round(size(csi_trace,1)*st):round(size(csi_trace,1)*en)
                if (mod(i,2) ~= 0)
                    continue;
                end
                csi_entry = csi_trace{i};
                if (size(csi_entry,2)>0) && (csi_entry.Nrx   == Nrx) && (csi_entry.Ntx == Ntx)
                    csi = get_scaled_csi(csi_entry(1,:,:));
                    rx_map=csi_entry.perm;
                    frq=1;
                    for t=1:Ntx
                        csi2=squeeze(csi(t,:,:));
                        % Sanitize ToFs with Algorithm 1
                        sanitized_csi = spotfi_algorithm_1(csi2,312.5e3);
                        tallmat=db(angle(sanitized_csi'));
                        for r=1:Nrx
                            for f=1:size(FREQ,2)
                                fp_csi(x+7,y+7,frq)=tallmat(FREQ(f),rx_map(r));
                                frq=frq+1;
                            end
                        end
                    end
                    %plot(db(abs(squeeze(csi2).')))
                end
            end
        catch exception
            disp('Error opening file');
            continue;
        end
    end
end

for ch=1:size(fp_csi,3)
    clear X;
    M=fp_csi(:,:,ch);
    M = [fliplr(M(:,floor(size(M,2)/2)+2:end)) M(:,floor(size(M,2)/2)+1:end)];
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
    fp_csi(:,:,ch)=O;
end

for ch=1:size(fp_csi,3)
    max_rgb=max(max(fp_csi(:,:,ch)));
    min_rgb=min(min(fp_csi(:,:,ch)));
    for x=-6:6
        for y=-6:6
            fp_csi_rgb(x+7,y+7,ch)=interp1([min_rgb,max_rgb],[0,255],fp_csi(x+7,y+7,ch));
            %rssi_rgb./max(rssi_rgb(:));
        end
    end
end
% Extract the individual red, green, and blue color channels.
redChannel = fp_csi_rgb(:, :, 1);
greenChannel = fp_csi_rgb(:, :, 2);
blueChannel = fp_csi_rgb(:, :, 3);
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
[X,Y] = meshgrid(1:13,1:13);
for ch=1:size(fp_csi,3)
    title(sprintf('Average CSI Phase values over Space 13m x 13m: Ch %d',ch),'FontSize',18)
    colorbar
    xlabel('X-axis in meters') 
    ylabel('Y-axis in meters')
    zlabel('CSI Phase values')
    Z = fp_csi(:,:,ch)';
    %C = X.*Y;
    surf(X,Y,Z)
    hold off;
    pause(1);
end