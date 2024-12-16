clear;
%csi_trace = read_bf_file('sample_data/log.all_csi.6.7.6');
%csi_trace = read_bf_file('sample_data/csi_local_ping.log');
%csi_trace = read_bf_file('sample_data/csi.dat');
%csi_trace = read_bf_file('sample_data/csi_ng_78mbps_stable.dat');
csi_trace = read_bf_file('sample_data/csi_ng_78mbps_stable2.dat');
csi_entry = csi_trace{1};
csi = get_scaled_csi(csi_entry(1,:,:));
csi3 = get_scaled_csi(csi_entry);
csi2 = zeros(1,3,30);
csi2=csi(1,:,:);
plot(db(abs(squeeze(csi2).')))
legend('RX Antenna A', 'RX Antenna B', 'RX Antenna C', 'Location', 'SouthEast' );
xlabel('Subcarrier index');
ylabel('SNR [dB]');
disp('Effective SNR for single antenna A');
db(get_eff_SNRs(csi2), 'pow') % for single antenna A

disp('Effective SNR using all available antenna');
db(get_eff_SNRs(csi3), 'pow')