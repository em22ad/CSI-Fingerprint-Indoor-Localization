% Phase Transformation
function output_phase=sanitize_phase(input_phase)

x = unwrap(input_phase);
a = ( x(30)-x(1) )/30.0;
b = sum(x)/30.0;
range = [-28,-26,-24,-22,-20,-18,-16,-14,-12,-10,-8,-6,-4,-2,-1,1,3,5,7,9,11,13,15,17,19,21,23,25,27,28]; % WiFi subcarrier indices at which CSI is available at 40 MHz
%range = [-15:-1 , 1:15];
output_phase = x - (a*range + b*ones(1,30));