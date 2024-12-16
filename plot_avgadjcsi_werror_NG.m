clear;
close all;
%csi_trace = read_bf_file('sample_data/csi_local_ping.log');
%csi_trace = read_bf_file('../CSI_dataset/csi_ng_0_6.dat');
%csi_trace = read_bf_file('../CSI_dataset_test/csi_ng_0_6.dat');
csi_trace = read_bf_file('../CSI_dataset_test/03_29_19_1_10pm/csi_ng_0_6.dat');
%csi_trace = read_bf_file('../CSI_dataset_test/03_29_19_8_35pm/csi_ng_0_6.dat');

Ntx=3;
Nrx=3;
TX=3;
RX=[1 2];
csi2 = zeros(1,3,30);    
%figure('name','CSI SNR change over time');
%hold on;
freq_st=25; %0,0 6,12,23 // 0,6 7,16,25
freq_end=25;
st=0.0;%0.015;
en=1.0;%0.025;
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
    csi2=csi(TX,:,:);
    tallmat=db(abs(squeeze(csi2)'));
    for j=1:3
        csi_col_ser{TX,ant_ord(j)}=[csi_col_ser{TX,ant_ord(j)} tallmat(:,ant_ord(j))];
    end
    %plot(tallmat)
    %legend('RX Antenna A', 'RX Antenna B', 'RX Antenna C', 'Location', 'SouthEast' );
    %xlabel('Subcarrier index');
    %ylabel('SNR [dB]');
end
disp(sprintf('actual observations used:%d',obs_ctr))
%hold off;
%%
std_csi=zeros(30);
mean_csi=zeros(30);
ad_ch_csi_col=[];
for i=1:30
    temp1=csi_col_ser{TX,ant_ord(RX(1))}(i,:); 
    temp2=csi_col_ser{TX,ant_ord(RX(2))}(i,:);
    new_row=(temp1+temp2)/2.0;
    ad_ch_csi_col=[ad_ch_csi_col;new_row];
end
id=find(ad_ch_csi_col == -Inf);
ad_ch_csi_col(id)=25;
mu=mean(ad_ch_csi_col(:));
ad_ch_csi_col(id)=mu;
mxval=max(ad_ch_csi_col(:));
mnval=min(ad_ch_csi_col(:));

ad_ch_csi_col=ad_ch_csi_col-mu;
ad_ch_csi_col=ad_ch_csi_col/(mxval-mnval);

ad_ch_filt =zeros(30,size(ad_ch_csi_col,2));

for i=1:30
    std_csi(i)=std(ad_ch_csi_col(i,:));
    mean_csi(i)=mean(ad_ch_csi_col(i,:));
    
    std_thresh=2.0;
    
    idxs=find(ad_ch_csi_col(i,:)<=mean_csi(i)+std_thresh*std_csi(i));
    idxs2=find(ad_ch_csi_col(i,idxs)>=mean_csi(i)-std_thresh*std_csi(i));
    temp_mean=mean(ad_ch_csi_col(i,idxs2));
    mean_csi(i)=temp_mean;
    ad_ch_filt(i,:)=temp_mean;
    ad_ch_filt(i,idxs2)=ad_ch_csi_col(i,idxs2);
end
%%
figure('name','CSI SNR Observation series 3-D plot' );
hold on;
%size(csi_col_ser{TX,1},2)
for i=1:size(csi_col_ser{TX,ant_ord(1)},2)
    plot3(1:30,csi_col_ser{TX,ant_ord(1)}(:,i),repmat(i,30))%,'color',rand(1,3)')
end
hold off;

figure('name','CSI SNR Observation series 3-D plot - Avg b/w adjacent Antenna pairs' );
hold on;
for i=1:size(ad_ch_filt,2)
    idx=find(ad_ch_filt(:,i)<-300);
    if (size(idx,1)<1)
        plot3(1:30,ad_ch_filt(:,i),repmat(i,30))%,'color',rand(1,3)')
    end
end
%for i=1:size(ad_ch_csi_col,2)
%    plot3(1:30,ad_ch_csi_col(:,i),repmat(i,30))
%end
ylim([-0.6 0.6])
hold off;