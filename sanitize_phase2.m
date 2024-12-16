% Phase Transformation
function sample_csi_trace_sanitized=sanitize_phase2(input_phase)
%%
fc = 5.765e9; % center frequency
M = 3;    % number of rx antennas
fs = 20e6; % channel bandwidth
c = 3e8;  % speed of light
d = 2.6e-1;  % distance between adjacent antennas in the linear antenna array 
% dTx = 2.6e-2; 
%SubCarrInd = [-58,-54,-50,-46,-42,-38,-34,-30,-26,-22,-18,-14,-10,-6,-2,2,6,10,14,18,22,26,30,34,38,42,46,50,54,58]; % WiFi subcarrier indices at which CSI is available at 40 MHz
SubCarrInd = [-28,-26,-24,-22,-20,-18,-16,-14,-12,-10,-8,-6,-4,-2,-1,1,3,5,7,9,11,13,15,17,19,21,23,25,27,28]; % WiFi subcarrier indices at which CSI is available at 40 MHz
N = length(SubCarrInd); % number of subcarriers
% subCarrSize = 128;  % total number fo
fgap = 312.5e3; % frequency gap in Hz between successive subcarriers in WiFi
lambda = c/fc;  % wavelength
T = 1; % number of transmitter antennas

% ToF sanitization code (Algorithm 1 in SpotFi paper)
csi_plot = reshape(input_phase, N, M);
[PhsSlope, PhsCons] = removePhsSlope(csi_plot,M,SubCarrInd,N);
ToMult = exp(1i* (-PhsSlope*repmat(SubCarrInd(:),1,M) - PhsCons*ones(N,M) ));
csi_plot = csi_plot.*ToMult;
relChannel_noSlope = reshape(csi_plot, N, M, T);
sample_csi_trace_sanitized = relChannel_noSlope(:);