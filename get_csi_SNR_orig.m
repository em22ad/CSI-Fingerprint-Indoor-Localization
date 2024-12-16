clear;
%csi_trace = read_bf_file('sample_data/csi_local_ping.log');
%csi_trace = read_bf_file('../CSI_dataset/csi_ng_0_2.dat');
csi_trace = read_bf_file('../CSI_dataset_test/csi_ng_0_6.dat');

Ntx=3;
TX=3;
csi2 = zeros(1,3,30);    
figure('name','CSI SNR change over time');
hold on;
st=0.0;
en=0.3;
start_obs=1+round(size(csi_trace,1)*st);
end_obs=round(size(csi_trace,1)*en);
disp(sprintf('Start:%d End:%d',start_obs,end_obs))
csi_col_ser = cell(3,3);
for i=1+round(size(csi_trace,1)*st):round(size(csi_trace,1)*en)
    csi_entry = csi_trace{i};
    if (size(csi_entry,1) <= 0)
        disp(sprintf('actual observations used:%d',i))
        break;
    end
    csi = get_scaled_csi(csi_entry(1,:,:));
    if (size(csi,1) ~= Ntx)
        continue;
    end
    csi2=csi(TX,:,:);
    tallmat=db(abs(squeeze(csi2).'));
    csi_col_ser{TX,1}=[csi_col_ser{TX,1} tallmat(:,1)];
    csi_col_ser{TX,2}=[csi_col_ser{TX,2} tallmat(:,2)];
    csi_col_ser{TX,3}=[csi_col_ser{TX,3} tallmat(:,3)];
    plot(tallmat)
    legend('RX Antenna A', 'RX Antenna B', 'RX Antenna C', 'Location', 'SouthEast' );
    xlabel('Subcarrier index');
    ylabel('SNR [dB]');
end

std_csi=zeros(3,3,30);
mean_csi=zeros(3,3,30);
for i=1:30
    std_csi(TX,1,i)=std(csi_col_ser{TX,1}(i,:));
    std_csi(TX,2,i)=std(csi_col_ser{TX,2}(i,:));
    std_csi(TX,3,i)=std(csi_col_ser{TX,3}(i,:));
    %sd1=std(csi_col_ser{1,1}(i,:));
    %sd2=std(csi_col_ser{1,2}(i,:));
    %sd3=std(csi_col_ser{1,3}(i,:));
    mean_csi(TX,1,i)=mean(csi_col_ser{TX,1}(i,:));
    mean_csi(TX,2,i)=mean(csi_col_ser{TX,2}(i,:));
    mean_csi(TX,3,i)=mean(csi_col_ser{TX,3}(i,:));
    std_thresh=1.0;
    idx1=find(csi_col_ser{TX,1}(i,:)<=mean_csi(TX,1,i)+std_thresh*std_csi(TX,1,i));
    idx2=find(csi_col_ser{TX,2}(i,:)<=mean_csi(TX,2,i)+std_thresh*std_csi(TX,2,i));
    idx3=find(csi_col_ser{TX,3}(i,:)<=mean_csi(TX,3,i)+std_thresh*std_csi(TX,3,i));
    mean_csi(TX,1,i)=mean(csi_col_ser{TX,1}(i,idx1));
    mean_csi(TX,2,i)=mean(csi_col_ser{TX,2}(i,idx2));
    mean_csi(TX,3,i)=mean(csi_col_ser{TX,3}(i,idx3));
end
hold off;

figure('name','CSI SNR average and variance');
hold on;
x = [1:30;1:30;1:30];
y = [mean_csi(TX,1,:);mean_csi(TX,2,:);mean_csi(TX,3,:)]; 
err=[std_csi(TX,1,:);std_csi(TX,2,:);std_csi(TX,3,:)];
h1=errorbar(x(1,:),y(1,:),err(1,:),'r');
h2=errorbar(x(2,:),y(2,:),err(2,:),'g');
h3=errorbar(x(3,:),y(3,:),err(3,:),'b');
%plot(x(1,:),y(1,:),'b');
if (TX==1)
    legend('RX Antenna A, TX Antenna A','RX Antenna A, TX Antenna B','RX Antenna A, TX Antenna C');
elseif (TX==2)
    legend('RX Antenna B, TX Antenna A','RX Antenna B, TX Antenna B','RX Antenna B, TX Antenna C');
else
    legend('RX Antenna C, TX Antenna A','RX Antenna C, TX Antenna B','RX Antenna C, TX Antenna C');
end
xlabel('Subcarrier index');
ylabel('SNR [dB]');
hold off;
%disp('Effective SNR');
%db(get_eff_SNRs(csi2), 'pow')