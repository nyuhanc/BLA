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