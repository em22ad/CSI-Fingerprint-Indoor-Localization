function [time_stamp] = extract_time_packet(packet)
 % where packet is your CSI file
 csi_trace=read_bf_file(packet);   
 
 for a=1:size(csi_trace,1)          %where j is the total number packets in the CSI file structure
     % extract the low 32 bits of the NIC's 1 MHz clock
     % The code doesn't account for wrapping about every 4300 seconds, or 72 minutes. 
     if (size(csi_trace{a,1},1) <= 0)
        continue;
     end
     time(a)=csi_trace{a,1}.timestamp_low; 
 end

 time_diff=(diff(time))*10^-6; %--convert difference b/w time_stamps to seconds
 actualtime=[0,time_diff];     %--row vector consisting of timestamp differences in seconds
 sum(time_diff,2);
 time_stamp=cumsum(actualtime);
end