clear;
close all;
%csi_trace = read_bf_file('sample_data/log.all_csi.6.7.6');
%csi_trace = read_bf_file('sample_data/csi_local_ping.log');
%csi_trace = read_bf_file('sample_data/csi.dat');
%csi_trace = read_bf_file('sample_data/csi_ng_78mbps_stable.dat');
%csi_trace = read_bf_file('sample_data/csi_ng_78mbps_stable2.dat');
csi_trace = read_bf_file('sample_data/csi_ng_-6_6.dat');
csi2 = zeros(1,3,30);
TX_Antenna=3;
RX_Antenna=3; %1:3
%figure;
%hold on;
%for i=int16(size(csi_trace,1)/2):int16(size(csi_trace,1)/1)
%for i=1:int16(size(csi_trace,1)/2)

avg_over=100;
fp=zeros(TX_Antenna*6,RX_Antenna*5);
fp_rssi=zeros(6,RX_Antenna*5);
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

for i=1:avg_over %int16(size(csi_trace,1))
    csi_entry = csi_trace{i};
    if (size(csi_entry,2)>0) && (csi_entry.Nrx   == RX_Antenna) && (csi_entry.Ntx   == TX_Antenna)
        sprintf('%1.1f',csi_entry.timestamp_low/67978);    
        csi = get_scaled_csi(csi_entry(1,:,:));
        
        for tx=1:TX_Antenna
            for rx=1:RX_Antenna
                max_csi=max(max(abs(csi(tx,rx,:))));
                min_csi=min(min(abs(csi(tx,rx,:))));
                csi(tx,rx,:)=interp1([min_csi,max_csi],[0,255],abs(csi(tx,rx,:)));
            end
        end
        
        for tx=1:TX_Antenna
            ysh=(tx-1);
            for rx=1:RX_Antenna
                xsh=(rx-1);
                l=1;
                csi2=csi(tx,rx,:);
                for j=1:6
                    for k=1:5
                        fp(j+(6*ysh),k+(5*xsh))=fp(j+(6*ysh),k+(5*xsh))+(csi2(1,1,l));
                        l=l+1;
                    end
                end
            end
        end
        %plot(db(abs(squeeze(csi2).')))
    end
end

fp=fp./avg_over;
fp_rssi=fp_rssi./avg_over;

I = mat2gray(fp);
J = imresize(I, 2.0);
figure
imshow(J)
%imshow(J);

%I_rssi = mat2gray(fp_rssi);


%legend(sprintf('RX Antenna: %d RX Antenna: %d',TX_Antenna,RX_Antenna));
%xlabel('Subcarrier index');
%ylabel('SNR [dB]');
%hold off;