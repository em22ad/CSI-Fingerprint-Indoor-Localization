clear;
close all;
FigPathName='C:/Users/em22a/OneDrive/Desktop/comparison/phase/';
%DSPathName='C:/Users/em22a/Downloads/coarse_localization_esp/csi_preproc_16freq/';
DSPathName='C:/Users/em22a/Downloads/coarse_localization_esp/csi_preproc/';
%PathName='../CSI_dataset/'; ds_tag='0';
%PathName='../CSI_dataset_test/';ds_tag='3';
%PathName='../CSI_dataset_test/03_29_19_1_10pm/'; ds_tag='1';
%PathName='../CSI_dataset_test/03_29_19_8_35pm/'; ds_tag='2';
PathName='../CSI_dataset_test/04_13_19_7_15pm/'; ds_tag='4';
obs2write=150;
std_thresh=2.0; %tolerable noise threshold
FREQ=[1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30];
%FREQ=[4 5 10 14 15 20 21 22 23 24 25 26 27 28 29 30];
sz_freq=size(FREQ,2);
Ntx=3;
Nrx=3;
TX=[1 2 3];
%TX=[2];
RX=[1 2 3];
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
            rx_map=csi_entry.perm;
            obs_ctr=obs_ctr+1;
            for t=1:size(TX,2)
%                 csi2=csi(t,:,:);
%                 uncalib_phase=angle(squeeze(csi2)');
%                 calib_phase(:,1)=sanitize_phase(uncalib_phase(:,1)');
%                 calib_phase(:,2)=sanitize_phase(uncalib_phase(:,2)');
%                 calib_phase(:,3)=sanitize_phase(uncalib_phase(:,3)');
%                 tallmat=db(calib_phase);
                
                csi2=squeeze(csi(t,:,:));
                % Sanitize ToFs with Algorithm 1
                sanitized_csi = spotfi_algorithm_1(csi2,312.5e3);
%                 uncalib_phase=angle(sanitized_csi');
%                 calib_phase(:,1)=sanitize_phase(uncalib_phase(:,1)');
%                 calib_phase(:,2)=sanitize_phase(uncalib_phase(:,2)');
%                 calib_phase(:,3)=sanitize_phase(uncalib_phase(:,3)');
%                 tallmat=db(calib_phase);
                tallmat=db(angle(sanitized_csi'));
                for j=1:3
                    csi_col_ser{t,j}=[csi_col_ser{t,j} tallmat(:,rx_map(j))];
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
                    new_row=csi_col_ser{t,RX(1,r)}(i,:);    
                    ad_ch_csi_col{t,r}=[ad_ch_csi_col{t,r};new_row];
                end
                id=find(ad_ch_csi_col{t,r} == Inf);
                ad_ch_csi_col{t,r}(id)=nan;
                ad_ch_csi_col{t,r}(id)=max(ad_ch_csi_col{t,r}(:));
                id=find(ad_ch_csi_col{t,r} == -Inf);
                ad_ch_csi_col{t,r}(id)=nan;
                ad_ch_csi_col{t,r}(id)=min(ad_ch_csi_col{t,r}(:));
                mu=nanmean(ad_ch_csi_col{t,r}(:));
                
                mxval=max(ad_ch_csi_col{t,r}(:));
                mnval=min(ad_ch_csi_col{t,r}(:));
                ad_ch_csi_col{t,r}=(ad_ch_csi_col{t,r}-mu)/(mxval-mnval);
            end
        end

        for t=1:size(TX,2)
            for r=1:size(RX,2)
                for i=1:sz_freq
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
        
        h1=figure('name','CSI Phase Time series 3-D plot - Filtered');
        n=1;
        sgtitle(sprintf('Location:(%d,%d)',xx,yy));
        for t=1:size(TX,2)
            for r=1:size(RX,2)
                subplot(3,3,n)
                n=n+1;
                obs_sz=size(ad_ch_csi_col{t,r},2);
%                 hold on;
%                 for i=1:sz_freq
%                     plot(obs_times(1:2:2*obs_sz),ad_ch_csi_col{t,r}(i,:))
%                 end
%                 xlabel('Time in secs');
%                 ylabel('SNR [dB]');
%                 title(sprintf('TX:%d, RX:%d',t,RX(1,r)));
%                 %ylim([-60.0 60.0])
%                 ylim([-0.6 0.6])
%                 hold off;  
                hold on;
                %size(csi_col_ser{TX,1},2)
                for i=1:size(ad_ch_csi_col{t,r},2)
                    %plot3(1:30,ad_ch_csi_col{t,r}(:,i),repmat(i,30))
                    plot(1:30,ad_ch_csi_col{t,r}(:,i))
                end
                ylim([-0.6 0.6])
                hold off;
            end
        end
        
        save_path=strcat(FigPathName,char(sprintf('%d%d_%s.png',xx,yy,ds_tag)));
        saveas(h1,save_path);
        
        final_op_mat=[];%cell(min(obs2write,obs_ctr),(size(TX,2)*size(RX,2)*sz_freq)+2);
%         final_op_mat(:,1)=num2cell(obs_times(1:2:min(obs2write,obs_ctr)*2)');
%         final_op_mat(:,2)=cellstr(sprintf('%d %d',xx,yy));
        row2=1;
        for i=1:min(obs2write,obs_ctr)
            wide_row=[];
            discard=0;
            for t=1:size(TX,2)
                for r=1:size(RX,2)
                    obs_sz=size(ad_ch_csi_col{t,r},2);
                    if(i > obs_sz)    
                        discard=1;
                    else
                        wide_row=[wide_row ad_ch_csi_col{t,r}(:,i)'];
                    end
                    
                    %else
                    %    disp(['The ',sprintf('TX:%d, RX:%d-%d',t,RX{1,r}(1),RX{1,r}(2)),' does not have ',num2str(obs2write),' obs.']);
                    %end
                end
            end
            if (discard == 0)
                new_row=[num2cell(obs_times(row2)) cellstr(sprintf('%d %d',xx,yy)) num2cell(wide_row)];  
                final_op_mat=[final_op_mat;new_row];
                row2=row2+2;
            end
        end


        
        overall_path=strcat(DSPathName,char(FileName(1)));
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
