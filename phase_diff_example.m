clear, clc, close all

% signal parameters
fs = 44100;
f0 = 1000;
T = 0.1;

% preparation of the time vector
N = round(T*fs);
t = (0:N-1)/fs;

% generation of the signal
x = sin(2*pi*f0*t) + 0.02*randn(1, N);
y = 0.5*sign(sin(2*pi*f0*t - pi/6)) + 0.02*randn(1, N);

% phase difference calculation
PhDiff = phdiffmeasure(x, y);
PhDiff = PhDiff*180/pi;

% display the phase difference
PhDiffstr = num2str(PhDiff);
disp(['Phase difference Y->X = ' PhDiffstr ' deg'])

% plot the signals
figure(1)
plot(t, x, 'b', 'LineWidth', 2)
grid on
hold on
plot(t, y, 'r', 'LineWidth', 2)
xlim([0 0.005])
ylim([-1.1 1.1])
set(gca, 'FontName', 'Times New Roman', 'FontSize', 14)
xlabel('Time, s')
ylabel('Amplitude, V')
title('Two signals with phase difference')
legend('First signal', 'Second signal')

commandwindow