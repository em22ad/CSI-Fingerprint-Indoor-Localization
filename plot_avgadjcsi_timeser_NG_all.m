clear;
close all;
%csi_trace = read_bf_file('../CSI_dataset/csi_ng_0_6.dat');
%csi_trace = read_bf_file('../CSI_dataset_test/csi_ng_0_6.dat');%02-28-4:30pm
%csi_trace = read_bf_file('../CSI_dataset_test/03_29_19_1_10pm/csi_ng_0_6.dat');
%csi_trace = read_bf_file('../CSI_dataset_test/03_29_19_8_35pm/csi_ng_0_6.dat');
%PathName='../CSI_dataset/';
%PathName='../CSI_dataset_test/';
%PathName='../CSI_dataset_test/03_29_19_1_10pm/';
PathName='../CSI_dataset_test/03_29_19_8_35pm/';

xx=-2;yy=6;
FileName{1}=sprintf('csi_ng_%d_%d.dat',xx,yy);
csi_trace = read_bf_file(strcat(PathName,char(FileName(1))));

obs_times = extract_time_packet(strcat(PathName,char(FileName(1))));
obs2write=50;
freq_st=1;
freq_end=30;
Ntx=3;
Nrx=3;
%TX=[1 2 3];
TX=[1];
RX{1,1}=[1 2];
RX{1,2}=[2 3];
RX{1,3}=[1 3];
csi2 = zeros(1,3,30);    
%figure('name','CSI SNR change over time');
%hold on;
st=0.0;%0.015;
en=1.0;%0.025;
start_obs=1+round(size(csi_trace,1)*st);
end_obs=round(size(csi_trace,1)*en);
disp(sprintf('Start:%d End:%d',start_obs,end_obs));
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
        tallmat=db(abs(squeeze(csi2)'));
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
            temp1=csi_col_ser{t,RX{1,r}(1)}(i,:);    
            temp2=csi_col_ser{t,RX{1,r}(2)}(i,:);
            
            new_row=(temp1+temp2)/2.0;
            ad_ch_csi_col{t,r}=[ad_ch_csi_col{t,r};new_row];
        end
        id=find(ad_ch_csi_col{t,r} == -Inf);
        ad_ch_csi_col{t,r}(id)=nan;
        mu=nanmean(ad_ch_csi_col{t,r}(:));
        ad_ch_csi_col{t,r}(id)=mu;
        mxval=max(ad_ch_csi_col{t,r}(:));
        mnval=min(ad_ch_csi_col{t,r}(:));
        ad_ch_csi_col{t,r}=(ad_ch_csi_col{t,r}-mu)/(mxval-mnval);
    end
end

ad_ch_filt=cell(3,3);
for t=1:size(TX,2)
    for r=1:size(RX,2)
        ad_ch_filt{t,r}=zeros(30,size(ad_ch_csi_col{t,r},2))-999;
    end
end

for t=1:size(TX,2)
    for r=1:size(RX,2)
        for i=1:30
            std_csi=nanstd(ad_ch_csi_col{t,r}(i,:));
            mean_csi=nanmean(ad_ch_csi_col{t,r}(i,:));

            std_thresh=2.0;     
            %if (t==3) && (r==1) && (i == 8)
            %    mean_csi
            %end
            
            idxs=find(ad_ch_csi_col{t,r}(i,:)<=mean_csi+std_thresh*std_csi);
            idxs2=find(ad_ch_csi_col{t,r}(i,idxs)>=mean_csi-std_thresh*std_csi);
            temp_mean=nanmean(ad_ch_csi_col{t,r}(i,idxs2));
            sz1=size(ad_ch_filt{t,r}(i,:),1);
            sz2=size(ad_ch_filt{t,r}(i,:),2);
            a = -0.5;
            b = 0.5;
            rnd = (b-a).*rand(sz1,sz2) + a;
            ad_ch_filt{t,r}(i,:)=repmat(mean_csi,sz1,sz2)+rnd*std_csi;
            %mean_csi(i)=temp_mean;
            ad_ch_filt{t,r}(i,idxs2)=ad_ch_csi_col{t,r}(i,idxs2);
        end
    end
end
%%
% h1=figure('name','CSI SNR Observation series 3-D plot' );
% 
% n=1;
% %size(csi_col_ser{TX,1},2)
% for t=1:size(TX,2)
%     for r=1:size(RX,2)
%         subplot(3,3,n)
%         n=n+1;
%         hold on;
%         for i=1:size(csi_col_ser{t,r},2)
%             plot3(1:30,csi_col_ser{t,r}(:,i),repmat(i,30))%,'color',rand(1,3)')
%         end
%         title(sprintf('TX:%d, RX:%d-%d',t,RX{1,r}(1),RX{1,r}(2)));
%         ylim([0 45])
%         hold off;
%     end
% end


% h2=figure('name','CSI SNR Observation series 3-D plot - Avg b/w adjacent Antenna pairs' );
% n=1;
% for t=1:size(TX,2)
%     for r=1:size(RX,2)
%         subplot(3,3,n)
%         n=n+1;
%         hold on;
%         for i=1:size(ad_ch_filt{t,r},2)
%             %idx=find(ad_ch_filt{t,r}(:,i)<-300);
%             %if (size(idx,1)<1)
%                 plot3(1:30,ad_ch_filt{t,r}(:,i),repmat(i,30))%,'color',rand(1,3)')
%             %end
%         end
%         title(sprintf('TX:%d, RX:%d-%d',t,RX{1,r}(1),RX{1,r}(2)));
%         ylim([-0.6 0.6])
%         hold off;
%     end
% end

figure('name','CSI SNR Time series 3-D plot');
n=1;
for t=1:size(TX,2)
    for r=1:size(RX,2)
        subplot(3,3,n)
        n=n+1;
        obs_sz=size(ad_ch_filt{t,r},2);
        hold on;
        for i=freq_st:freq_end
            plot(obs_times(1:2:2*obs_sz),ad_ch_filt{t,r}(i,:))
        end
        xlabel('Time in secs');
        ylabel('SNR [dB]');
        title(sprintf('TX:%d, RX:%d-%d',t,RX{1,r}(1),RX{1,r}(2)));
        ylim([-0.5 0.5])
        hold off;
    end
end

final_op_mat=cell(obs2write,(size(TX,2)*size(RX,2)*30)+2);
final_op_mat(:,1)=num2cell(obs_times(1:2:obs2write*2)');
final_op_mat(:,2)=cellstr(sprintf('%d %d',xx,yy));
row=1;
for i=1:min(obs2write,obs_ctr)
    wide_row=[];
    for t=1:size(TX,2)
        for r=1:size(RX,2)
            obs_sz=size(ad_ch_filt{t,r},2);
            if(obs_sz >=min(obs2write,obs_ctr))
                wide_row=[wide_row ad_ch_filt{t,r}(:,i)'];
            else
                disp(['The ',sprintf('TX:%d, RX:%d-%d',t,RX{1,r}(1),RX{1,r}(2)),' does not have ',num2str(min(obs2write,obs_ctr)),' obs.']);
            end
        end
    end
    final_op_mat(row,3:end)=num2cell(wide_row);
    row=row+1;
end
overall_path=strcat(PathName,char(FileName(1)));
overall_path=sprintf('%scsv',overall_path(1:size(overall_path,2)-3)); 
% fid = fopen(overall_path,'wt');
% if fid>0
%     for k=1:size(final_op_mat,1)
%         fprintf(fid,'%f\n',final_op_mat{k,3:end});
%     end
%     fclose(fid);
% end
fid = fopen(overall_path,'wt');
if fid>0
    for k=1:size(final_op_mat,1)
        fprintf(fid,'%d\n',k);
    end
    fclose(fid);
end
xlswrite(overall_path,final_op_mat);
%saveas(h1,sprintf('raw.png'))
%saveas(h2,sprintf('proc.png'))
%for i=1:size(ad_ch_csi_col,2)
%    plot3(1:30,ad_ch_csi_col(:,i),repmat(i,30))
%end
%ylim([0 35])

%disp('Effective SNR');
%db(get_eff_SNRs(csi2), 'pow')