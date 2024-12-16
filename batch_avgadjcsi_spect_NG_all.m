clear;
close all;
%csi_trace = read_bf_file('sample_data/csi_local_ping.log');
%csi_trace = read_bf_file('../CSI_dataset/csi_ng_0_6.dat');
%csi_trace = read_bf_file('../CSI_dataset_test/csi_ng_0_6.dat');%02-28-4:30pm
%csi_trace = read_bf_file('../CSI_dataset_test/03_29_19_1_10pm/csi_ng_0_6.dat');
%csi_trace = read_bf_file('../CSI_dataset_test/03_29_19_8_35pm/csi_ng_0_6.dat');
%PathName='../CSI_dataset/'; ds_tag='0';
PathName='../CSI_dataset_test/';ds_tag='3';
%PathName='../CSI_dataset_test/03_29_19_1_10pm/'; ds_tag='1';
%PathName='../CSI_dataset_test/03_29_19_8_35pm/'; ds_tag='2';
%PathName='../CSI_dataset_test/04_13_19_7_15pm/'; ds_tag='4';
obs2write=150;
std_thresh=1.0; %tolerable noise threshold
FREQ=[1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30];
%FREQ=[4 5 10 14 15 20 21 22 23 24 25 26 27 28 29 30];
sz_freq=size(FREQ,2);
Ntx=3;
Nrx=3;
TX=[1 2 3];
%TX=[2];
RX{1,1}=[1 2];
RX{1,2}=[2 3];
RX{1,3}=[1 3];
for xx=-6:2:6
    for yy=-6:2:6
        FileName{1}=sprintf('csi_ng_%d_%d.dat',xx,yy);
        if ~exist(strcat(PathName,char(FileName(1))), 'file')
            errorMessage=sprintf('File: %s not found.',strcat(PathName,char(FileName(1))));
            disp(errorMessage);
            continue;
        end
        csi_trace = read_bf_file(strcat(PathName,char(FileName(1))));

        obs_times = extract_time_packet(strcat(PathName,char(FileName(1))));

        csi2 = zeros(1,3,30);    
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
        end
        disp(sprintf('actual observations used:%d',obs_ctr))
        if (obs_ctr <= 0)
            disp('Skipping');
            continue;
        end
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
                ad_ch_filt{t,r}=zeros(sz_freq,size(ad_ch_csi_col{t,r},2))-999;
            end
        end

        for t=1:size(TX,2)
            for r=1:size(RX,2)
                for i=1:sz_freq
                    std_csi=nanstd(ad_ch_csi_col{t,r}(FREQ(i),:));
                    mean_csi=nanmean(ad_ch_csi_col{t,r}(FREQ(i),:));
     
                    %if (t==3) && (r==1) && (i == 8)
                    %    mean_csi
                    %end

                    idxs=find(ad_ch_csi_col{t,r}(FREQ(i),:)<=mean_csi+std_thresh*std_csi);
                    idxs2=find(ad_ch_csi_col{t,r}(FREQ(i),idxs)>=mean_csi-std_thresh*std_csi);
                    temp_mean=nanmean(ad_ch_csi_col{t,r}(FREQ(i),idxs2));
                    sz1=size(ad_ch_filt{t,r}(i,:),1);
                    sz2=size(ad_ch_filt{t,r}(i,:),2);
                    a = -0.5;
                    b = 0.5;
                    rnd = (b-a).*rand(sz1,sz2) + a;
                    ad_ch_filt{t,r}(i,:)=repmat(mean_csi,sz1,sz2)+rnd*std_csi;
                    %mean_csi(i)=temp_mean;
                    ad_ch_filt{t,r}(i,idxs2)=ad_ch_csi_col{t,r}(FREQ(i),idxs2);
                end
            end
        end
        
        figure('name','CSI SNR Time series 3-D plot - Filtered');
        n=1;
        for t=1:size(TX,2)
            for r=1:size(RX,2)
                subplot(3,3,n)
                n=n+1;
                obs_sz=size(ad_ch_filt{t,r},2);
                hold on;
                for i=1:sz_freq
                    plot(obs_times(1:2:2*obs_sz),ad_ch_filt{t,r}(i,:))
                end
                xlabel('Time in secs');
                ylabel('SNR [dB]');
                title(sprintf('TX:%d, RX:%d-%d',t,RX{1,r}(1),RX{1,r}(2)));
                ylim([-0.5 0.5])
                hold off;
            end
        end

        final_op_mat=cell(min(obs2write,obs_ctr),(size(TX,2)*size(RX,2)*sz_freq)+2);
        final_op_mat(:,1)=num2cell(obs_times(1:2:min(obs2write,obs_ctr)*2)');
        final_op_mat(:,2)=cellstr(sprintf('%d %d',xx,yy));
        row=1;
        for i=1:min(obs2write,obs_ctr)
            wide_row=[];
            for t=1:size(TX,2)
                for r=1:size(RX,2)
                    obs_sz=size(ad_ch_filt{t,r},2);
                    %if(obs_sz >=obs2write)
                        wide_row=[wide_row ad_ch_filt{t,r}(:,i)'];
                    %else
                    %    disp(['The ',sprintf('TX:%d, RX:%d-%d',t,RX{1,r}(1),RX{1,r}(2)),' does not have ',num2str(obs2write),' obs.']);
                    %end
                end
            end
            final_op_mat(row,3:end)=num2cell(wide_row);
            row=row+1;
        end
        overall_path=strcat(PathName,char(FileName(1)));
        overall_path=sprintf('%s_%s.csv',overall_path(1:size(overall_path,2)-4),ds_tag); 
        
        fid = fopen(overall_path,'wt');
        if fid>0
            for k=1:size(final_op_mat,1)
                fprintf(fid,'%d\n',k);
            end
            fclose(fid);
        end
        xlswrite(overall_path,final_op_mat);
    end
end
