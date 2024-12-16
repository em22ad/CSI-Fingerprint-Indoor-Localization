clear;
close all;
FigPathName='C:/Users/em22a/OneDrive/Desktop/comparison/phase/';
%DSPathName='C:/Users/em22a/Downloads/coarse_localization_esp/csi_preproc_phasediff/';
DSPathName='C:/Users/em22a/Downloads/coarse_localization_esp/csi_preproc_phasedir_1tx/';
%PathName='../CSI_dataset/'; ds_tag='0';
%PathName='../CSI_dataset_test/';ds_tag='3';
%PathName='../CSI_dataset_test/03_29_19_1_10pm/'; ds_tag='1';
%PathName='../CSI_dataset_test/03_29_19_8_35pm/'; ds_tag='2';
PathName='../CSI_dataset_test/04_13_19_7_15pm/'; ds_tag='4';
obs2write=6; %or 6 %ignoring 1,3 pair for all TX
std_thresh=2.0; %tolerable noise threshold
FREQ=[1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30];
%FREQ=[4 5 10 14 15 20 21 22 23 24 25 26 27 28 29 30];
sz_freq=size(FREQ,2);
Ntx=3;
Nrx=3;
TX=[2];
%TX=[2];
RX=[1 2 3];

ad_ch_csi_col=cell(13,13);
res_x=2;
res_y=2;
for xx=-6:res_x:6
    for yy=-6:res_y:6
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
            %rx_map=[1 2 3];
            obs_ctr=obs_ctr+1;
            for t=1:size(TX,2)
                csi2=squeeze(csi(TX(t),:,:));
                % Sanitize ToFs with Algorithm 1
                sanitized_csi = spotfi_algorithm_1(csi2,312.5e3);
                % Acquire smoothed CSI matrix
%                 smoothed_sanitized_csi = smooth_csi(sanitized_csi);
%                 uncalib_phase=angle(csi2');
%                 calib_phase(:,1)=sanitize_phase(uncalib_phase(:,1)');
%                 calib_phase(:,2)=sanitize_phase(uncalib_phase(:,2)');
%                 calib_phase(:,3)=sanitize_phase(uncalib_phase(:,3)');
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
        
        ad_ch_csi_col{xx+7,yy+7}=[];
       
        new_cols=[]; %6 Phase Diff (1 for each antenna pair for each frequency
        for t=1:size(TX,2)
           for r=1:size(RX,2)
               for i=1:30
                    col=csi_col_ser{t,RX(r)}(i,:)';   

                    id=find(col == -Inf);
                    col(id)=nan;
                    col(id)=min(col(:));

                    id=find(col == Inf);
                    col(id)=nan;
                    col(id)=max(col(:));

                    mu=nanmean(col(:));
                    mxval=max(col(:));
                    mnval=min(col(:));
                    col=col-mu;
                    col=col./(mxval-mnval);
                    id=find(isnan(col));
                    if (size(id,1) > 0)
                        disp('Nan found');
                    end
                    new_cols=[new_cols col];
               end
           end
        end
        ad_ch_csi_col{xx+7,yy+7}=[ad_ch_csi_col{xx+7,yy+7};new_cols];
   end
end
%%
%total_moves=13*((13-1)/2)*2;
mark_done=zeros(13,13);
for xx=-6:res_x:6
    for yy=-6:res_y:6
        if (size(ad_ch_csi_col{xx+7,yy+7},1) > 0)
            move_fp=[];%cell(1,total_moves);
            nbr_list=find_nbr(ad_ch_csi_col,xx+7,yy+7,res_x,res_y);
            for j=1:size(nbr_list,1)
                if (mark_done(nbr_list(j,1),nbr_list(j,2)) == 0)
                    temp1=ad_ch_csi_col{xx+7,yy+7};    
                    temp2=ad_ch_csi_col{nbr_list(j,1),nbr_list(j,2)};
                    obs_under_consd=min(size(temp1,1),size(temp2,1));
                    temp1=temp1(1:obs_under_consd,:);
                    temp2=temp2(1:obs_under_consd,:);
                    %mark_done(nbr_list(j,1),nbr_list(j,2))=1;
                    move_fp=[move_fp;repmat(cellstr(sprintf('%d %d %d %d',xx,yy,nbr_list(j,1)-7,nbr_list(j,2)-7)),obs_under_consd,1)  num2cell(temp2)  num2cell(temp1-temp2)];
                    move_fp=[move_fp;repmat(cellstr(sprintf('%d %d %d %d',nbr_list(j,1)-7,nbr_list(j,2)-7,xx,yy)),obs_under_consd,1)  num2cell(temp1)  num2cell(temp2-temp1)];
%                     move_fp=[move_fp;repmat(cellstr(sprintf('%d %d %d %d',xx,yy,nbr_list(j,1)-7,nbr_list(j,2)-7)),obs_under_consd,1)  num2cell(temp2)  num2cell(temp1)];
%                     move_fp=[move_fp;repmat(cellstr(sprintf('%d %d %d %d',nbr_list(j,1)-7,nbr_list(j,2)-7,xx,yy)),obs_under_consd,1)  num2cell(temp1)  num2cell(temp2)];
                end
            end
            mark_done(xx+7,yy+7)=1;
            if (size(move_fp,1) > 0)              
                FileName{1}=sprintf('csi_ng_%d_%d.dat',xx,yy);
                overall_path=strcat(DSPathName,char(FileName(1)));
                overall_path=sprintf('%s_%s.csv',overall_path(1:size(overall_path,2)-4),ds_tag);

                fid = fopen(overall_path,'wt');
                if fid>0
                    for k=1:size(move_fp,1)
                        fprintf(fid,'%d\n',k);
                    end
                    fclose(fid);
                end
                xlswrite(overall_path,move_fp);
            end
        end
    end
end
%%
 



