import numpy
import scipy.io
import scipy.signal
import scipy.fft

import matplotlib.pyplot as plt

### CONTROL PANEL (for the vocoder (it's just a vocoder))
output_sample_rate = 16000
# segment time to segment spacing ratio affects synthesized pitch
segment_time_ms = 15
segment_spacing_ms = 10  # start-to-start
filter_bandwidth_octaves = 4 / 12 # 4 semitones
# ISO Frequency bands, good for general purpose, maybe not ideal for vocal signals
# filter_bank_nominal_frequencies = [16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000]
# filter_bank_nominal_frequencies = [400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300]

input_window_function = scipy.signal.windows.hann
output_window_function = scipy.signal.windows.hann
# https://docs.scipy.org/doc/scipy/reference/signal.windows.html
# window_function = scipy.signal.windows.hamming
# window_function = scipy.signal.windows.boxcar
# window_function = scipy.signal.windows.barthann
# window_function = scipy.signal.windows.blackman
# window_function = scipy.signal.windows.blackmanharris
# window_function = scipy.signal.windows.bohman
# window_function = scipy.signal.windows.cosine
# window_function = scipy.signal.windows.flattop
# window_function = scipy.signal.windows.lanczos
# window_function = scipy.signal.windows.nuttall
# window_function = scipy.signal.windows.parzen
# window_function = scipy.signal.windows.taylor
# window_function = scipy.signal.windows.triang
# window_function = scipy.signal.windows.tukey
###

fig, ax = plt.subplots(3, 3)
fig.tight_layout()

# Load the stream from the .wav into your program
sample_rate, sample_array = scipy.io.wavfile.read("input.wav")

# If it is a stereo recording (2 channels), keep only one of them and discard the other
sample_array = sample_array[:,0]

# I like float type samples, let's convert
if numpy.issubdtype(sample_array.dtype, numpy.integer):
    min = numpy.iinfo(sample_array.dtype).min
    max = numpy.iinfo(sample_array.dtype).max
    mid = (min + max) // 2
    sample_array = sample_array.astype('float')
    sample_array -= mid
    sample_array /= max

# If the sampling rate is > 16KHz, down-sample to 16kHz
sample_array = scipy.signal.resample_poly(sample_array, output_sample_rate, sample_rate)
sample_rate = output_sample_rate

# find frequency bands for vocal signals by using fast Fourier transform
fastFT = abs(scipy.fft.fft(sample_array))
ax[2, 0].plot(fastFT)
ax[2, 0].title.set_text("FFT Analysis")
ax[2, 0].set_xlabel('Frequency (Hz)')
ax[2, 0].set_ylabel('Magnitude (dB)')
ax[2, 0].grid(True)
filter_bank_nominal_frequencies = [410, 800, 1350, 2700, 3484, 3582, 3720, 3858, 4072, 4104, 4123] # chose strong frequencies making up the input wave

ax[0, 0].set_xlabel('Time (ms)')
ax[0, 0].set_ylabel('Amplitude')
ax[0, 0].title.set_text("Input Downsampled Waveform")
ax[0, 0].set_ylim([-1, 1.1])
ax[0, 0].plot(numpy.arange(len(sample_array)) / sample_rate * 1000, sample_array, 'r', )

# Divide the stream into successive short ”chunks”
# Lets define a helper function to convert ms to sample count
def ms_to_samples(ms):
    return sample_rate * ms // 1000

segment_length = ms_to_samples(segment_time_ms)
segment_spacing = ms_to_samples(segment_spacing_ms)

segment_count = len(sample_array) - segment_length // segment_spacing + 1

# Lets create a window function for our segments
# A hanning function will probally sound the best but we can also use rectangular or whatever
input_segment_window = input_window_function(segment_length)
output_segment_window = output_window_function(segment_length)

ax[1, 1].title.set_text('Input Segment Window Function')
ax[1, 1].set_xlabel('Time (ms)')
ax[1, 1].set_ylabel('Amplitude')
ax[1, 1].set_ylim([0, 1.1])
ax[1, 1].plot(numpy.arange(len(input_segment_window)) / sample_rate * 1000, input_segment_window)

ax[1, 2].title.set_text('Output Segment Window Function')
ax[1, 2].set_xlabel('Time (ms)')
ax[1, 2].set_ylabel('Amplitude')
ax[1, 2].set_ylim([0, 1.1])
ax[1, 2].plot(numpy.arange(len(output_segment_window)) / sample_rate * 1000, output_segment_window)

# Slice up the sample array into segments and apply the window
audio_segments = []
for i in range(0, segment_count, segment_spacing):
    samples = sample_array[i:i+segment_length]
    segment = input_segment_window * numpy.pad(samples, (0, segment_length - len(samples)))
    audio_segments.append(segment)

# Helper function for RMS
def RMS(x):
   return numpy.sqrt(numpy.mean(x ** 2))

# Filter bank generation
filter_bank = []
bandwidth = filter_bandwidth_octaves
for nominal in filter_bank_nominal_frequencies:
    # Test for nyquist
    if nominal >= sample_rate / 2:
        continue
    frequency_low = nominal / (2 ** (bandwidth / 2))
    frequency_high = nominal * (2 ** (bandwidth / 2))
    bandpass_filter = scipy.signal.butter(2, [frequency_low, frequency_high], btype='bandpass', output='sos', fs=sample_rate)
    filter_bank.append((nominal, bandpass_filter))
    frequencies, response = scipy.signal.sosfreqz(bandpass_filter, fs=sample_rate, worN=2048)
    ax[1, 0].plot(frequencies, 20 * numpy.log10(numpy.abs(response)))

ax[1, 0].title.set_text('Filter Bank Response')
ax[1, 0].set_xlabel('Frequency (Hz)')
ax[1, 0].set_ylabel('Magnitude (dB)')
ax[1, 0].set_ylim([-12, 1])
ax[1, 0].set_xlim([filter_bank_nominal_frequencies[0] / (2 ** bandwidth), filter_bank_nominal_frequencies[-1] * (2 ** bandwidth) ])
ax[1, 0].grid()

# Filter each segment detect RMS and synthesize sin waves then reassemble segment
output_sample_count = len(audio_segments) * segment_spacing - segment_spacing + segment_length # After segment re-assembly (expected segment at the end of the function)
output_samples = numpy.zeros(output_sample_count)
for i, segment in enumerate(audio_segments):
    composite_sine_segment = numpy.zeros(len(segment))
    for frequency, filter in filter_bank:
        time_domain = numpy.arange(0, segment_length) / sample_rate
        sine_signal = numpy.sin(2 * numpy.pi * frequency * time_domain)
        rms = RMS(scipy.signal.sosfilt(filter, segment))
        composite_sine_segment += rms * sine_signal * output_segment_window # Use the window function again (It makes it smoother (if your segments overlap))
    padding_start = segment_spacing * i
    padding_end = len(output_samples) - segment_length - padding_start
    output_samples += numpy.pad(composite_sine_segment, (padding_start, padding_end))

ax[0, 1].set_xlabel('Time (ms)')
ax[0, 1].set_ylabel('Amplitude')
ax[0, 1].title.set_text("Vocoded Waveform (pre-normalization)")
ax[0, 1].set_ylim([-1, 1.1])
ax[0, 1].plot(numpy.arange(len(output_samples)) / sample_rate * 1000, output_samples, 'r')

# Normalize
output_samples /= numpy.max(numpy.abs(output_samples))


ax[0, 2].set_xlabel('Time (ms)')
ax[0, 2].set_ylabel('Amplitude')
ax[0, 2].title.set_text("Vocoded Waveform (post-normalization)")
ax[0, 2].set_ylim([-1, 1.1])
ax[0, 2].plot(numpy.arange(len(output_samples)) / sample_rate * 1000, output_samples, 'r', )

scipy.io.wavfile.write("output.wav", sample_rate, output_samples.astype('float32'))

plt.show()