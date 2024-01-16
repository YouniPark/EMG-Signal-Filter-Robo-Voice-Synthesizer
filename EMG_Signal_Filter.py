#!/usr/bin/env python
# coding: utf-8

# ## Filter to Remove Power Line Interference 

# #### Second Order Butterworth Notch-Filter (Band-Stop) with BW = 5 Hz, fc1 = 57.5 Hz, fc2 = 62.5 Hz. 
# #### Goal is to remove the 60Hz interference from the 0.1 to 450Hz signal. 

# In[4]:


import pandas as pd
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Read EMG data from the CSV file
file_path = 'C:/Users/kulsu/Downloads/EMG_Datasets.csv'
df = pd.read_csv(file_path)

# Assuming your EMG data is in columns 'Time (s)', 'EMG_Relaxed (mV)', and 'EMG_Contracted (mV)'
time = df['Time (s)'].values
emg_relaxed = df['EMG_Relaxed (mV)'].values
emg_contracted = df['EMG_Contracted (mV)'].values

# Design the notch filter to remove 60Hz noise with a 5 Hz bandwidth
order = 2
f0 = 60            # Center frequency to be removed
bw = 5             # Bandwidth
f1 = f0 - bw / 2   #Cutoff Frequencies
f2 = f0 + bw / 2
sos = signal.butter(N=order, Wn=[f1,f2], btype='bandstop', fs=1000, output='sos')

# Apply the notch filter to the EMG data
emg_relaxed_filtered = signal.sosfilt(sos, emg_relaxed)
emg_contracted_filtered = signal.sosfilt(sos, emg_contracted)

# Compute the FFT for both unfiltered and filtered EMG data
N = len(emg_relaxed)
xf = fftfreq(N, 1 / 1000)  # ts=1/fs is 0.001 (1000 Hz)
xf = xf[0:N // 2]
emg_relaxed_fft = (2 / N) * np.abs(fft(emg_relaxed))
emg_relaxed_filtered_fft = (2 / N) * np.abs(fft(emg_relaxed_filtered))

emg_contracted_fft = (2 / N) * np.abs(fft(emg_contracted))
emg_contracted_filtered_fft = (2 / N) * np.abs(fft(emg_contracted_filtered))

# Plot the frequency spectra for the Relaxed EMG data
plt.figure(figsize=(10, 6))
plt.plot(xf, emg_relaxed_fft[0:N // 2], '--', label='Relaxed EMG (Original)')
plt.plot(xf, emg_relaxed_filtered_fft[0:N // 2], 'k', label='Relaxed EMG (Filtered)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Frequency Spectrum of Relaxed EMG Data')
plt.legend()

# Plot the frequency spectra for the Contracted EMG data
plt.figure(figsize=(10, 6))
plt.plot(xf, emg_contracted_fft[0:N // 2], '--', label='Contracted EMG (Original)')
plt.plot(xf, emg_contracted_filtered_fft[0:N // 2], 'k', label='Contracted EMG (Filtered)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Frequency Spectrum of Contracted EMG Data')
plt.legend()

# Plot the time domain signals for Relaxed EMG data
plt.figure(figsize=(10, 6))
plt.plot(time, emg_relaxed, label='Relaxed EMG (Original)')
plt.plot(time, emg_relaxed_filtered, 'k', label='Relaxed EMG (Filtered)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (mV)')
plt.title('Time Domain of Relaxed EMG Data')
plt.legend()

# Plot the time domain signals for Contracted EMG data
plt.figure(figsize=(10, 6))
plt.plot(time, emg_contracted, label='Contracted EMG (Original)')
plt.plot(time, emg_contracted_filtered, 'k', label='Contracted EMG (Filtered)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (mV)')
plt.title('Time Domain of Contracted EMG Data')
plt.legend()


# ## Filter to Remove Noise outside of 0.1 Hz & 450 Hz Range

# #### Second Order Butterworth BandPass Filter with BW = 449.5, fc1 = 0.1 Hz, fc2 = 450 Hz. 
# #### Goal is to remove any noise outsde of the 0.1 to 450Hz signal.

# In[5]:


import pandas as pd
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Read EMG data from the CSV file
file_path = 'C:/Users/kulsu/Downloads/EMG_Datasets.csv'
df = pd.read_csv(file_path)

# Assuming your EMG data is in columns 'Time (s)', 'EMG_Relaxed (mV)', and 'EMG_Contracted (mV)'
time = df['Time (s)'].values
emg_relaxed = df['EMG_Relaxed (mV)'].values
emg_contracted = df['EMG_Contracted (mV)'].values

# Design the bandpass filter with cutoff frequencies 0.1 Hz and 450 Hz
order = 2
f1 = 0.1
f2 = 450
sos = signal.butter(N=order, Wn=[f1,f2], btype='bandpass', fs=1000, output='sos')

# Apply the bandpass filter
emg_relaxed_filtered = signal.sosfilt(sos, emg_relaxed)
emg_contracted_filtered = signal.sosfilt(sos, emg_contracted)

# Compute the FFT for both unfiltered and filtered EMG data
N = len(emg_relaxed)
xf = fftfreq(N, 1 / 1000)
xf = xf[0:N // 2]
emg_relaxed_fft = (2 / N) * np.abs(fft(emg_relaxed))
emg_relaxed_filtered_fft = (2 / N) * np.abs(fft(emg_relaxed_filtered))

emg_contracted_fft = (2 / N) * np.abs(fft(emg_contracted))
emg_contracted_filtered_fft = (2 / N) * np.abs(fft(emg_contracted_filtered))

# Plot the frequency spectra for the Relaxed EMG data
plt.figure(figsize=(10, 6))
plt.plot(xf, emg_relaxed_fft[0:N // 2], '--', label='Relaxed EMG (Original)')
plt.plot(xf, emg_relaxed_filtered_fft[0:N // 2], 'k', label='Relaxed EMG (Filtered)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Frequency Spectrum of Relaxed EMG Data')
plt.legend()
plt.grid(True)
plt.show()

# Plot the frequency spectra for the Contracted EMG data
plt.figure(figsize=(10, 6))
plt.plot(xf, emg_contracted_fft[0:N // 2], '--', label='Contracted EMG (Original)')
plt.plot(xf, emg_contracted_filtered_fft[0:N // 2], 'k', label='Contracted EMG (Filtered)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Frequency Spectrum of Contracted EMG Data')
plt.legend()
plt.grid(True)
plt.show()

# Plot the time domain signals for Relaxed EMG data
plt.figure(figsize=(10, 6))
plt.plot(time, emg_relaxed, label='Relaxed EMG (Original)')
plt.plot(time, emg_relaxed_filtered, 'k', label='Relaxed EMG (Filtered)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (mV)')
plt.title('Time Domain of Relaxed EMG Data')
plt.legend()
plt.grid(True)
plt.show()

# Plot the time domain signals for Contracted EMG data
plt.figure(figsize=(10, 6))
plt.plot(time, emg_contracted, label='Contracted EMG (Original)')
plt.plot(time, emg_contracted_filtered, 'k', label='Contracted EMG (Filtered)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (mV)')
plt.title('Time Domain of Contracted EMG Data')
plt.legend()
plt.grid(True)
plt.show()




# ## Filter to Deal with Both Interference & Noise 

# ### Add RMS of each signal

# In[6]:


import pandas as pd
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Read EMG data from the CSV file
file_path = 'C:/Users/kulsu/Downloads/EMG_Datasets.csv'
df = pd.read_csv(file_path)

# Assuming your EMG data is in columns 'Time (s)', 'EMG_Relaxed (mV)', and 'EMG_Contracted (mV)'
time = df['Time (s)'].values
emg_relaxed = df['EMG_Relaxed (mV)'].values
emg_contracted = df['EMG_Contracted (mV)'].values

# Design the notch filter to remove 60Hz noise with a 5 Hz bandwidth
order_notch = 2
f0 = 60           # Center frequency to be removed
bw = 5            # Bandwidth

f1_notch = f0 - bw / 2  # Cutoff Frequencies
f2_notch = f0 + bw / 2
sos_notch = signal.butter(N=order_notch, Wn=[f1_notch, f2_notch], btype='bandstop', fs=1000, output='sos')

# Apply the notch filter to the EMG data
emg_relaxed_notch = signal.sosfilt(sos_notch, emg_relaxed)
emg_contracted_notch = signal.sosfilt(sos_notch, emg_contracted)

# Design the bandpass filter with cutoff frequencies 0.1 Hz and 450 Hz
order_bp = 2
f1_bp = 0.1
f2_bp = 450
sos_bp = signal.butter(N=order_bp, Wn=[f1_bp, f2_bp], btype='bandpass', fs=1000, output='sos')

# Apply the bandpass filter to the previously notch filtered EMG data
emg_relaxed_filtered = signal.sosfilt(sos_bp, emg_relaxed_notch)
emg_contracted_filtered = signal.sosfilt(sos_bp, emg_contracted_notch)

# Compute the FFT for both unfiltered and filtered EMG data
N = len(emg_relaxed)
xf = fftfreq(N, 1 / 1000)
xf = xf[0:N // 2]
emg_relaxed_fft = (2 / N) * np.abs(fft(emg_relaxed))
emg_relaxed_filtered_fft = (2 / N) * np.abs(fft(emg_relaxed_filtered))

emg_contracted_fft = (2 / N) * np.abs(fft(emg_contracted))
emg_contracted_filtered_fft = (2 / N) * np.abs(fft(emg_contracted_filtered))

# Plot the frequency spectra for the Relaxed EMG data
plt.figure(figsize=(10, 6))
plt.plot(xf, emg_relaxed_fft[0:N // 2], '--', label='Relaxed EMG (Original)')
plt.plot(xf, emg_relaxed_filtered_fft[0:N // 2], 'k', label='Relaxed EMG (Filtered)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Frequency Spectrum of Relaxed EMG Data')
plt.legend()
plt.grid(True)
plt.show()

# Plot the frequency spectra for the Contracted EMG data
plt.figure(figsize=(10, 6))
plt.plot(xf, emg_contracted_fft[0:N // 2], '--', label='Contracted EMG (Original)')
plt.plot(xf, emg_contracted_filtered_fft[0:N // 2], 'k', label='Contracted EMG (Filtered)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Frequency Spectrum of Contracted EMG Data')
plt.legend()
plt.grid(True)
plt.show()

# Plot the time domain signals for Relaxed EMG data
plt.figure(figsize=(10, 6))
plt.plot(time, emg_relaxed, label='Relaxed EMG (Original)')
plt.plot(time, emg_relaxed_filtered, 'k', label='Relaxed EMG (Filtered)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (mV)')
plt.title('Time Domain of Relaxed EMG Data')
plt.legend()
plt.grid(True)
plt.show()

# Plot the time domain signals for Contracted EMG data
plt.figure(figsize=(10, 6))
plt.plot(time, emg_contracted, label='Contracted EMG (Original)')
plt.plot(time, emg_contracted_filtered, 'k', label='Contracted EMG (Filtered)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (mV)')
plt.title('Time Domain of Contracted EMG Data')
plt.legend()
plt.grid(True)
plt.show()

# Calculate RMS of each signal before and after filtering using loops
def calculate_rms(samples):
    sum_of_squares = np.sum(samples ** 2)
    rms = np.sqrt(sum_of_squares / len(samples))
    return rms

rms_relaxed_original = calculate_rms(emg_relaxed)
rms_relaxed_filtered = calculate_rms(emg_relaxed_filtered)

rms_contracted_original = calculate_rms(emg_contracted)
rms_contracted_filtered = calculate_rms(emg_contracted_filtered)

print(f"RMS of Relaxed EMG (Original): {rms_relaxed_original:.2f} mV")
print(f"RMS of Relaxed EMG (Filtered): {rms_relaxed_filtered:.2f} mV")

print(f"RMS of Contracted EMG (Original): {rms_contracted_original:.2f} mV")
print(f"RMS of Contracted EMG (Filtered): {rms_contracted_filtered:.2f} mV")


# In[ ]:




