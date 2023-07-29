#!/usr/bin/env python
# coding: utf-8

# ## Filter to Remove Power Line Interference 

# #### Second Order Notch-Filter (Band-Stop) with BW = 5Hz, fc1 = 58 Hz, fc2 = 63 Hz. 
# #### Goal is to remove the 60Hz interference from the 0.1 to 450Hz signal.

# In[106]:


import pandas as pd
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# Step 1: Read EMG data from the CSV file
file_path = 'C:/Users/kulsu/Downloads/EMG_Datasets.csv'
df = pd.read_csv(file_path)

# Assuming EMG data is in columns 'Time (s)', 'EMG_Relaxed (mV)', and 'EMG_Contracted (mV)'
time = df['Time (s)'].values
emg_relaxed = df['EMG_Relaxed (mV)'].values
emg_contracted = df['EMG_Contracted (mV)'].values

# Step 2: Design the notch filter
fs = 1000  # Replace with your actual sampling frequency
BW = 5  # Bandwidth in Hz
notch_freq = 60  # Notch frequency to remove (60 Hz)

# Normalize the notch frequency based on the sampling frequency
normalized_notch_freq = notch_freq / (fs / 2)

# Calculate the Q-factor of the notch filter
Q = normalized_notch_freq / BW

# Design the notch filter using the bilinear transform
b, a = signal.iirnotch(normalized_notch_freq, Q)

# Step 3: Apply the notch filter to the EMG signals
filtered_emg_relaxed = signal.lfilter(b, a, emg_relaxed)
filtered_emg_contracted = signal.lfilter(b, a, emg_contracted)

# Step 4: Compute the FFT of the original and filtered EMG signals
freq_relaxed, fft_relaxed = signal.welch(emg_relaxed, fs, nperseg=1024)
freq_filtered_relaxed, fft_filtered_relaxed = signal.welch(filtered_emg_relaxed, fs, nperseg=1024)

freq_contracted, fft_contracted = signal.welch(emg_contracted, fs, nperseg=1024)
freq_filtered_contracted, fft_filtered_contracted = signal.welch(filtered_emg_contracted, fs, nperseg=1024)

# Step 5: Plot the FFT
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.semilogy(freq_relaxed, fft_relaxed, label='Original EMG (Relaxed)')
plt.semilogy(freq_filtered_relaxed, fft_filtered_relaxed, label='Filtered EMG (Relaxed)')
plt.title('FFT of EMG Signal (Relaxed)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid()
plt.legend()

plt.subplot(2, 1, 2)
plt.semilogy(freq_contracted, fft_contracted, label='Original EMG (Contracted)')
plt.semilogy(freq_filtered_contracted, fft_filtered_contracted, label='Filtered EMG (Contracted)')
plt.title('FFT of EMG Signal (Contracted)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()


# In[107]:


plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(time, emg_relaxed, label='Original EMG (Relaxed)')
plt.plot(time, filtered_emg_relaxed, label='Filtered EMG (Relaxed)')
plt.title('EMG Signal (Relaxed)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (mV)')
plt.grid()
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(time, emg_contracted, label='Original EMG (Contracted)')
plt.plot(time, filtered_emg_contracted, label='Filtered EMG (Contracted)')
plt.title('EMG Signal (Contracted)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (mV)')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()


# In[ ]:




