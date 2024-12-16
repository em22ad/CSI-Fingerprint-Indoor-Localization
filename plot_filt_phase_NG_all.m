clear;
close all;
%csi_trace = read_bf_file('sample_data/csi_local_ping.log');
csi_trace = read_bf_file('../CSI_dataset/csi_ng_0_0.dat');
%csi_trace = read_bf_file('../CSI_dataset_test/csi_ng_-4_6.dat');%02-28-4:30pm
%csi_trace = read_bf_file('../CSI_dataset_test/03_29_19_1_10pm/csi_ng_-4_6.dat');
%csi_trace = read_bf_file('../CSI_dataset_test/03_29_19_8_35pm/csi_ng_-4_6.dat');
%obs_times=extract_time_packet('../CSI_dataset_test/03_29_19_8_35pm/csi_ng_0_6.dat')

std_thresh=2.0; %noise threshold
Ntx=3;
Nrx=3;
TX=[1 2 3];
RX=[1 2 3];
csi2 = zeros(1,3,30);    
%figure('name','CSI SNR change over time');
%hold on;
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
    obs_ctr=obs_ctr+1;
    for t=1:size(TX,2)
        csi2=csi(t,:,:);
        uncalib_phase=angle(squeeze(csi2)'); 
        calib_phase(:,1)=phase_calibration(uncalib_phase(:,1));
        calib_phase(:,2)=phase_calibration(uncalib_phase(:,2));
        calib_phase(:,3)=phase_calibration(uncalib_phase(:,3));
        tallmat=db(calib_phase);
        for j=1:3
            csi_col_ser{t,j}=[csi_col_ser{t,j} tallmat(:,j)];
        end
    end
    %plot(tallmat)
    %legend('RX Antenna A', 'RX Antenna B', 'RX Antenna C', 'Location', 'SouthEast' );
    %xlabel('Subcarrier index');
    %ylabel('SNR [dB]');
end
disp(sprintf('actual observations used:%d',obs_ctr))
%hold off;
%%
ad_ch_csi_col=cell(3,3);
for t=1:size(TX,2)
    for r=1:size(RX,2)
        ad_ch_csi_col{t,r}=[];
        for i=1:30
            new_row=csi_col_ser{t,RX(1,r)}(i,:);    
            ad_ch_csi_col{t,r}=[ad_ch_csi_col{t,r};new_row];
        end
        id=find(ad_ch_csi_col{t,r} == -Inf);
        ad_ch_csi_col{t,r}(id)=nan;
        id=find(ad_ch_csi_col{t,r} == Inf);
        ad_ch_csi_col{t,r}(id)=nan;
        mu=nanmean(ad_ch_csi_col{t,r}(:));
        ad_ch_csi_col{t,r}(id)=mu;
        mxval=max(ad_ch_csi_col{t,r}(:));
        mnval=min(ad_ch_csi_col{t,r}(:));
        ad_ch_csi_col{t,r}=(ad_ch_csi_col{t,r}-mu)/(mxval-mnval);
    end
end

for t=1:size(TX,2)
    for r=1:size(RX,2)
        for i=1:30
            std_csi=nanstd(ad_ch_csi_col{t,r}(i,:));
            mean_csi=nanmean(ad_ch_csi_col{t,r}(i,:));
     
            %if (t==3) && (r==1) && (i == 8)
            %    mean_csi
            %end
            
            idxs=find(ad_ch_csi_col{t,r}(i,:)>=mean_csi+std_thresh*std_csi);
            ad_ch_csi_col{t,r}(:,idxs)=[];
            idxs2=find(ad_ch_csi_col{t,r}(i,:)<=mean_csi-std_thresh*std_csi);
            ad_ch_csi_col{t,r}(:,idxs2)=[];
        end
    end
end
%%
h1=figure('name','CSI SNR Time series 3-D plot' );
n=1;
%size(csi_col_ser{TX,1},2)
for t=1:size(TX,2)
    for r=1:size(RX,2)
        subplot(3,3,n)
        n=n+1;
        hold on;
        for i=1:size(ad_ch_csi_col{t,r},2)
            plot3(1:30,ad_ch_csi_col{t,r}(:,i),repmat(i,30))%,'color',rand(1,3)')
        end
        title(sprintf('TX:%d, RX:%d',t,RX(1,r)));
        %ylim([0 45])
        ylim([-0.6 0.6])
        hold off;
    end
end

%saveas(h1,sprintf('raw.png'))
%saveas(h2,sprintf('proc.png'))
%for i=1:size(ad_ch_csi_col,2)
%    plot3(1:30,ad_ch_csi_col(:,i),repmat(i,30))
%end
%ylim([0 35])


%disp('Effective SNR');
%db(get_eff_SNRs(csi2), 'pow')