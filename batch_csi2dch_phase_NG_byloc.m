clear;
close all;
FigPathName='C:/Users/em22a/OneDrive/Desktop/comparison/phase/';
%DSPathName='C:/Users/em22a/Downloads/coarse_localization_esp/csi_preproc_16freq/';
DSPathName='C:/Users/em22a/Downloads/coarse_localization_esp/csi_preproc/';
PathName='../CSI_dataset/'; ds_tag='0';INTERPOLATE_NODATA_POINTS=1;std_thresh=2.5; %tolerable noise threshold
%PathName='../CSI_dataset_test/';ds_tag='3';INTERPOLATE_NODATA_POINTS=0;std_thresh=2.5; %tolerable noise threshold
%PathName='../CSI_dataset_test/03_29_19_1_10pm/'; ds_tag='1';INTERPOLATE_NODATA_POINTS=0;std_thresh=3.0; %tolerable noise threshold
%PathName='../CSI_dataset_test/03_29_19_8_35pm/'; ds_tag='2';INTERPOLATE_NODATA_POINTS=1;std_thresh=3.0; %tolerable noise threshold
%PathName='../CSI_dataset_test/04_13_19_7_15pm/'; ds_tag='4';INTERPOLATE_NODATA_POINTS=0;std_thresh=2.5; %tolerable noise threshold
obs2write=150;
FREQ=[1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30];
%FREQ=[4 5 10 14 15 20 21 22 23 24 25 26 27 28 29 30];
sz_freq=size(FREQ,2);
TX=[1 2 3];
RX=[1 2 3];
Ntx=size(TX,2);
Nrx=size(RX,2);
MAX_OBS=500;
NO_READING=-2.0;
x_range=[-3 2];
y_range=[0 3];
x_diff=x_range(2)-x_range(1);
y_diff=y_range(2)-y_range(1);
fp_csi=zeros(MAX_OBS,x_diff+1,y_diff+1,Ntx*Nrx*sz_freq)+NO_READING;
max_obs_rec=zeros(x_diff+1,y_diff+1)+MAX_OBS;

for xx=x_range(1):1:x_range(2)
    for yy=y_range(1):1:y_range(2)
        FileName{1}=sprintf('csi_ng_%d_%d.dat',xx*2,yy*2);
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
        fprintf('Start:%d End:%d\n',start_obs,end_obs);
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
            %rx_map=csi_entry.perm;
            rx_map=[1 2 3];
            obs_ctr=obs_ctr+1;
            for t=1:size(TX,2)          
                csi2=squeeze(csi(TX(t),:,:));
                % Sanitize ToFs with Algorithm 1
                sanitized_csi = spotfi_algorithm_1(csi2,312.5e3);
                tallmat=db(angle(sanitized_csi'));
                for j=1:size(RX,2)
                    csi_col_ser{TX(t),RX(j)}=[csi_col_ser{TX(t),RX(j)} tallmat(:,rx_map(j))];
                end
            end
        end
        
        fprintf('actual observations used:%d\n',obs_ctr)
        if (obs_ctr <= 0)
            disp('Skipping');
            continue;
        end
        ad_ch_csi_col=cell(3,3);
        for t=1:size(TX,2)
            for r=1:size(RX,2)
                ad_ch_csi_col{TX(t),RX(r)}=[];
                for i=1:sz_freq
                    new_row=csi_col_ser{TX(t),RX(r)}(FREQ(i),:);    
                    ad_ch_csi_col{TX(t),RX(r)}=[ad_ch_csi_col{TX(t),RX(r)};new_row];
                end
                id=find(ad_ch_csi_col{TX(t),RX(r)} == Inf);
                ad_ch_csi_col{TX(t),RX(r)}(id)=nan;
                ad_ch_csi_col{TX(t),RX(r)}(id)=max(ad_ch_csi_col{TX(t),RX(r)}(:));
                id=find(ad_ch_csi_col{TX(t),RX(r)} == -Inf);
                ad_ch_csi_col{TX(t),RX(r)}(id)=nan;
                ad_ch_csi_col{TX(t),RX(r)}(id)=min(ad_ch_csi_col{TX(t),RX(r)}(:));
                mu=nanmean(ad_ch_csi_col{TX(t),RX(r)}(:));
                
                mxval=max(ad_ch_csi_col{TX(t),RX(r)}(:));
                mnval=min(ad_ch_csi_col{TX(t),RX(r)}(:));
                ad_ch_csi_col{TX(t),RX(r)}=(ad_ch_csi_col{TX(t),RX(r)}-mu)/(mxval-mnval);
            end
        end

        for t=1:size(TX,2)
            for r=1:size(RX,2)
                for i=1:sz_freq
                    std_csi=nanstd(ad_ch_csi_col{TX(t),RX(r)}(FREQ(i),:));
                    
                    data=squeeze(ad_ch_csi_col{TX(t),RX(r)}(FREQ(i),:));
                    sz=length(data);
                    subset=floor(sz/2);
                    for i2=1:subset
                       x2(i2) = median(data(randsample(sz,floor(subset/2))'));
                    end
                    mean_csi = nanmean(x2);
                    %mean_csi=nanmean(ad_ch_csi_col{TX(t),RX(r)}(FREQ(i),:)); %take a the median of a random mean rather than a full mean as there is a chance of missing out the noisy points
                    idxs=find(ad_ch_csi_col{TX(t),RX(r)}(FREQ(i),:)>=mean_csi+std_thresh*std_csi); %think about making this criteria more stringent for test datasets
                    ad_ch_csi_col{TX(t),RX(r)}(:,idxs)=[]; % delete the noisy obervations
                    idxs2=find(ad_ch_csi_col{TX(t),RX(r)}(FREQ(i),:)<=mean_csi-std_thresh*std_csi);
                    ad_ch_csi_col{TX(t),RX(r)}(:,idxs2)=[]; % delete the noisy obervations
                end
            end
        end
        
        frq=1;
        min_obs_for_loc=100000;
        for t=1:size(TX,2)
            for r=1:size(RX,2)
                for i=1:sz_freq
                    if (min_obs_for_loc > size(ad_ch_csi_col{TX(t),RX(r)},2))
                        min_obs_for_loc=size(ad_ch_csi_col{TX(t),RX(r)},2);
                    end
                    for obs=1:size(ad_ch_csi_col{TX(t),RX(r)},2)
                        fp_csi(obs,xx+4,yy+1,frq)=ad_ch_csi_col{TX(t),RX(r)}(FREQ(i),obs);
                    end
                    frq=frq+1;
                end
            end
        end
        max_obs_rec(xx+4,yy+1)=min_obs_for_loc;
    end
end
%%
for i=1:min(max_obs_rec(:))
    fp=squeeze(fp_csi(i,:,:,:));
    if (INTERPOLATE_NODATA_POINTS == 1)
        fp=interp_missing_pixels(fp);
    end
%    fp=compact_fp(fp);
%     figure
%     [X,Y] = meshgrid(1:x_diff,1:y_diff);
%     for ch=1:size(fp,3)
%         title(sprintf('Average CSI Phase values over Space %dm x %dm: Ch %d',x_diff, y_diff, ch),'FontSize',18)
%         colorbar
%         xlabel('X-axis in meters') 
%         ylabel('Y-axis in meters')
%         zlabel('CSI Phase values')
%         Z = fp(:,:,ch)';
%         %C = X.*Y;
%         surf(X,Y,Z)
%         hold off;
%         pause(1);
%     end

    overall_path=strcat(DSPathName,char(FileName(1)));
    overall_path=sprintf('%s_%s_obs%d.csv',overall_path(1:size(overall_path,2)-8),ds_tag,i); 

    fid = fopen(overall_path,'wt');
    if fid>0
        for k=1:size(fp,1)
            fprintf(fid,'%d\n',k);
        end
        fclose(fid);
    end
    xlswrite(overall_path,reshape(fp,size(fp,1)*size(fp,2),Ntx*Nrx*sz_freq));
end