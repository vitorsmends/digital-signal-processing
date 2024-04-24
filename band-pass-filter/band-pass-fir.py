import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter
from pydub import AudioSegment

def apply_bandpass_filter(signal, low_cutoff_freq, high_cutoff_freq, sample_rate, filter_order=1001):
    nyquist_rate = sample_rate / 2
    normalized_low_cutoff_freq = low_cutoff_freq / nyquist_rate
    normalized_high_cutoff_freq = high_cutoff_freq / nyquist_rate
    taps = firwin(filter_order, [normalized_low_cutoff_freq, normalized_high_cutoff_freq], pass_zero=False, window='hamming')
    filtered_signal = lfilter(taps, 1.0, signal)
    return filtered_signal.astype(np.int16), taps

def plot_signal_and_fft(input_signal, output_signal, sample_rate, title):
    plt.figure(figsize=(12, 6))
    time = np.arange(len(input_signal)) / sample_rate
    
    # Plot input signal
    plt.subplot(2, 1, 1)
    plt.plot(time, input_signal, label='Input Signal')
    plt.plot(time, output_signal, label='Output Signal')
    plt.title(title + ' - Temporal Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    
    # Calculate FFT for input signal
    fft_input_signal = np.fft.fft(input_signal)
    freqs = np.fft.fftfreq(len(input_signal), d=1/sample_rate)
    
    # Plot FFT for input signal
    plt.subplot(2, 1, 2)
    plt.plot(freqs[:len(freqs)//2], np.abs(fft_input_signal)[:len(freqs)//2], label='Input Signal')
    
    # Calculate FFT for output signal
    fft_output_signal = np.fft.fft(output_signal)
    
    # Plot FFT for output signal
    plt.plot(freqs[:len(freqs)//2], np.abs(fft_output_signal)[:len(freqs)//2], label='Output Signal')
    
    plt.title(title + ' - FFT')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.tight_layout()
    plt.show()

def apply_bandpass_filter_to_audio_file(input_file, output_file, low_cutoff_freq, high_cutoff_freq, filter_order=1001):
    audio = AudioSegment.from_file(input_file)
    sample_width = audio.sample_width
    sample_rate = audio.frame_rate
    channels = audio.channels
    signal = np.array(audio.get_array_of_samples())
    filtered_signal, taps = apply_bandpass_filter(signal, low_cutoff_freq, high_cutoff_freq, sample_rate, filter_order)
    plot_signal_and_fft(signal, filtered_signal, sample_rate, "Input/Output")
    filtered_audio = AudioSegment(filtered_signal.tobytes(), 
                                  frame_rate=sample_rate,
                                  sample_width=sample_width,
                                  channels=channels)
    filtered_audio.export(output_file, format="mp3")

# Example usage
input_file = "./input-data/input.mp3"
output_file = "./output-data/output_bandpass_fir.mp3"
low_cutoff_freq = 500  # Adjust low cutoff frequency as needed
high_cutoff_freq = 2000  # Adjust high cutoff frequency as needed
filter_order = 1001  # Adjust filter order as needed

apply_bandpass_filter_to_audio_file(input_file, output_file, low_cutoff_freq, high_cutoff_freq, filter_order)
