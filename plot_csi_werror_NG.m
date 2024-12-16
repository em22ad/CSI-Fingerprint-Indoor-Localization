clear;
close all;
%csi_trace = read_bf_file('sample_data/csi_local_ping.log');
%csi_trace = read_bf_file('../CSI_dataset/csi_ng_0_6.dat');
%csi_trace = read_bf_file('../CSI_dataset_test/csi_ng_0_6.dat');
%csi_trace = read_bf_file('../CSI_dataset_test/03_29_19_1_10pm/csi_ng_0_6.dat');
csi_trace = read_bf_file('../CSI_dataset_test/03_29_19_8_35pm/csi_ng_0_6.dat');
obs_times = extract_time_packet('../CSI_dataset_test/03_29_19_8_35pm/csi_ng_0_6.dat');

Ntx=3;
Nrx=3;
TX=2;
RX=1;
freq_st=1;
freq_end=30;
csi2 = zeros(1,3,30);
%figure('name','CSI SNR change over time');
%hold on;
st=0.0;%0.015;
en=1.0;%0.025;1
start_obs=1+round(size(csi_trace,1)*st);
end_obs=round(size(csi_trace,1)*en);
disp(sprintf('Start:%d End:%d',start_obs,end_obs))
csi_col_ser = cell(3,3);
obs_ctr=0;
for i=1+round(size(csi_trace,1)*st):round(size(csi_trace,1)*en)
    if (mod(i,2) ~= 0)
        continue;
    end
    
    csi_entry = csi_trace{i};
    
    if (size(csi_entry,1) <= 0)
        continue;
    end
    csi = get_scaled_csi(csi_entry(1,:,:));
    if (size(csi,1) < Ntx)
        continue;
    end
    ant_ord=[1 2 3];
    obs_ctr=obs_ctr+1;
    csi2=csi(ant_ord(TX),:,:);
    tallmat=db(abs(squeeze(csi2)'));
    for j=1:3
        csi_col_ser{ant_ord(TX),ant_ord(j)}=[csi_col_ser{ant_ord(TX),ant_ord(j)} tallmat(:,ant_ord(j))];
    end
    %plot(tallmat)
    %legend('RX Antenna A', 'RX Antenna B', 'RX Antenna C', 'Location', 'SouthEast' );
    %xlabel('Subcarrier index');
    %ylabel('SNR [dB]');
end
disp(sprintf('actual observations used:%d',obs_ctr))
%hold off;
%%
std_csi=zeros(3,3,30);
mean_csi=zeros(3,3,30);

for i=1:30
    for j=1:Nrx
        std_csi(ant_ord(TX),ant_ord(j),i)=std(csi_col_ser{ant_ord(TX),ant_ord(j)}(i,:));
    end
    
    for j=1:Nrx
        mean_csi(ant_ord(TX),ant_ord(j),i)=mean(csi_col_ser{ant_ord(TX),ant_ord(j)}(i,:));
    end
    
    std_thresh=2.0;
    for j=1:Nrx
        idxs=find(csi_col_ser{ant_ord(TX),ant_ord(j)}(i,:)<=mean_csi(ant_ord(TX),ant_ord(j),i)+std_thresh*std_csi(ant_ord(TX),ant_ord(j),i));
        idxs2=find(csi_col_ser{ant_ord(TX),ant_ord(j)}(i,idxs)>=mean_csi(ant_ord(TX),ant_ord(j),i)-std_thresh*std_csi(ant_ord(TX),ant_ord(j),i));
        mean_csi(ant_ord(TX),ant_ord(j),i)=mean(csi_col_ser{ant_ord(TX),ant_ord(j)}(i,idxs2));
    end
end
%%
figure('name','CSI SNR Observation series 3-D plot' );
hold on;
%size(csi_col_ser{TX,1},2)
for i=1:size(csi_col_ser{ant_ord(TX),ant_ord(RX)},2)
    plot3(1:30,csi_col_ser{ant_ord(TX),ant_ord(RX)}(:,i),repmat(i,30))
end
ylim([0 35])
hold off;

figure('name','CSI SNR Time series 3-D plot' );
hold on;
obs_sz=size(csi_col_ser{ant_ord(TX),ant_ord(RX)},2);
for i=freq_st:freq_end
    plot3(obs_times(1:2:2*obs_sz),csi_col_ser{ant_ord(TX),ant_ord(RX)}(i,:),repmat(i,1,obs_sz))
end
xlabel('Time in secs');
ylabel('SNR [dB]');
zlabel('Carrier index');
ylim([0 35])
hold off;

figure('name','CSI SNR average and variance');
hold on;
x = [1:30;1:30;1:30];
y = [mean_csi(ant_ord(TX),ant_ord(1),:);mean_csi(ant_ord(TX),ant_ord(2),:);mean_csi(ant_ord(TX),ant_ord(3),:)]; 
err=[std_csi(ant_ord(TX),ant_ord(1),:);std_csi(ant_ord(TX),ant_ord(2),:);std_csi(ant_ord(TX),ant_ord(3),:)];
h1=errorbar(x(ant_ord(1),:),y(ant_ord(1),:),err(ant_ord(1),:),'r');
h2=errorbar(x(ant_ord(2),:),y(ant_ord(2),:),err(ant_ord(2),:),'g');
h3=errorbar(x(ant_ord(3),:),y(ant_ord(3),:),err(ant_ord(3),:),'b');
%plot(x(1,:),y(1,:),'b');
if (ant_ord(TX)==1)
    legend('RX Antenna A, TX Antenna A','RX Antenna B, TX Antenna A','RX Antenna C, TX Antenna A');
elseif (ant_ord(TX)==2)
    legend('RX Antenna A, TX Antenna B','RX Antenna B, TX Antenna B','RX Antenna C, TX Antenna B');
else
    legend('RX Antenna A, TX Antenna C','RX Antenna B, TX Antenna C','RX Antenna C, TX Antenna C');
end
xlabel('Subcarrier index');
ylabel('SNR [dB]');
hold off;
%disp('Effective SNR');
%db(get_eff_SNRs(csi2), 'pow')