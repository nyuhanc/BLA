import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

# suppose `a` and `b` are your numpy arrays
a = np.random.random(1000) - 0.5
b = np.random.random(1000) - 0.5

# calculate cross-correlation
xcorr = scipy.signal.correlate(a, b, mode='full')

# calculate the number of overlapping points at each lag
overlap = np.correlate(np.ones_like(a), np.ones_like(b), mode='full')

# normalize cross-correlation by the overlap
xcorr_normalized = xcorr / overlap

# compute significance level
N = len(a)  # number of samples
stat_sig = 2 / np.sqrt(N)

# plot cross-correlation and significance level
lags = np.arange(-N + 1, N)  # array of lags
plt.plot(lags, xcorr_normalized)
plt.plot(lags, [stat_sig]*len(lags), 'r')
plt.plot(lags, [-stat_sig]*len(lags), 'r')
plt.show()







# filter out noise  with Butterworth filter
cutoff = 0.05
order = 10
fs=1

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

u1_b = butter_lowpass_filter(u1, cutoff, fs, order)

plt.subplot(2, 1, 2)
#plt.plot(np.arange(len(u1)), u1, 'b-', label='data')
plt.plot(np.arange(len(u1)), u1_b, 'g-', linewidth=2, label='filtered data')
plt.xlabel('Time [sec]')
plt.grid()
plt.legend()
plt.show()