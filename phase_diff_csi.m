clear, clc, close all
%%
PathName{1}='../CSI_dataset/csi_ng_0_0.dat';
PathName{2}='../CSI_dataset_test/csi_ng_0_0.dat';
PathName{3}='../CSI_dataset_test/03_29_19_1_10pm/csi_ng_0_0.dat';
PathName{4}='../CSI_dataset_test/03_29_19_8_35pm/csi_ng_0_0.dat';
Ntx=3;
Nrx=3;
TX=2;
RX=2;
freq_st=1;
freq_end=30;

% plot the signals
figure(1)
grid on
hold on

for ds=1:size(PathName,2)
    csi_trace = read_bf_file(PathName{ds});
    t = extract_time_packet(PathName{ds});
    idx=(1:2:size(t,2));
    t=t(idx);
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
        ant_ord=csi_entry.perm;
        %ant_ord=[1 2 3];
        obs_ctr=obs_ctr+1;
%         csi2=csi(TX,:,:);
%         uncalib_phase=angle(squeeze(csi2)');
%         calib_phase(:,1)=sanitize_phase(uncalib_phase(:,1)');
%         calib_phase(:,2)=sanitize_phase(uncalib_phase(:,2)');
%         calib_phase(:,3)=sanitize_phase(uncalib_phase(:,3)');
%         tallmat=db(calib_phase);
        csi2=squeeze(csi(TX,:,:));
        % Sanitize ToFs with Algorithm 1
        sanitized_csi = spotfi_algorithm_1(csi2,312.5e3);
        tallmat=db(angle(sanitized_csi'));

        for j=1:3
            csi_col_ser{TX,j}=[csi_col_ser{TX,j} tallmat(:,ant_ord(j))];
        end
        %plot(tallmat)
        %legend('RX Antenna A', 'RX Antenna B', 'RX Antenna C', 'Location', 'SouthEast' );
        %xlabel('Subcarrier index');
        %ylabel('SNR [dB]');
    end

    disp(sprintf('actual observations used:%d',obs_ctr))
    t=t(1:obs_ctr);
    %%
    % phase difference calculation
    phdiff_12=[];
    phdiff_23=[];

    for i=1:30
        x=csi_col_ser{TX,1}(i,:);
        y=csi_col_ser{TX,2}(i,:);
        PhDiff = phdiffmeasure(x,y);
        PhDiff = PhDiff*180/pi;
        if (isnan(PhDiff))
            PhDiff=0;%mean(phdiff_12);
        end
        phdiff_12=[phdiff_12 abs(PhDiff)];
        

        x=csi_col_ser{TX,2}(i,:);
        y=csi_col_ser{TX,3}(i,:);
        PhDiff = phdiffmeasure(x,y);
        PhDiff = PhDiff*180/pi;
        if (isnan(PhDiff))
            PhDiff=0;%mean(phdiff_12);
        end
        
        phdiff_23=[phdiff_23 abs(PhDiff)];

        % display the phase difference
        %PhDiffstr = num2str(PhDiff);
        %disp(['Phase difference Y->X = ' PhDiffstr ' deg'])
    end
    mu=mean(phdiff_12);
    phdiff_12=phdiff_12-mu;
     
    mu=mean(phdiff_23);
    phdiff_23=phdiff_23-mu;
    
    plot(1:30, phdiff_12, 'color',rand(1,3), 'LineWidth', 2)
    %plot(1:30, phdiff_23, 'color',rand(1,3), 'LineWidth', 2)
end
hold off
set(gca, 'FontName', 'Times New Roman', 'FontSize', 14)
xlabel('Freq Carrier #')
ylabel('Phase Difference (Radians), delta r')
title('Phase difference over frequencies over antenna pair 1-2 over locations')
%legend('Antenna 1-2', 'Antenna 2-3')
legend('Antenna 1-2')
% commandwindow