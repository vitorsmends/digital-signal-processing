import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import iirfilter, lfilter
from pydub import AudioSegment

def apply_lowpass_filter(signal, cutoff_freq, sample_rate, filter_order=4):
    nyquist_rate = sample_rate / 2
    normalized_cutoff_freq = cutoff_freq / nyquist_rate
    b, a = iirfilter(filter_order, normalized_cutoff_freq, btype='low', analog=False, ftype='butter')
    filtered_signal = lfilter(b, a, signal)
    return filtered_signal.astype(np.int16), b, a

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

def apply_lowpass_filter_to_audio_file(input_file, output_file, cutoff_freq, filter_order=4):
    audio = AudioSegment.from_file(input_file)
    sample_width = audio.sample_width
    sample_rate = audio.frame_rate
    channels = audio.channels
    signal = np.array(audio.get_array_of_samples())
    filtered_signal, b, a = apply_lowpass_filter(signal, cutoff_freq, sample_rate, filter_order)
    plot_signal_and_fft(signal, filtered_signal, sample_rate, "Input/Output")
    filtered_audio = AudioSegment(filtered_signal.tobytes(), 
                                  frame_rate=sample_rate,
                                  sample_width=sample_width,
                                  channels=channels)
    filtered_audio.export(output_file, format="mp3")

# Example usage
input_file = "./input-data/input.mp3"
output_file = "./output-data/output_lowpass_iir.mp3"
cutoff_freq = 1000  # Adjust cutoff frequency as needed
filter_order = 4  # Adjust filter order as needed

apply_lowpass_filter_to_audio_file(input_file, output_file, cutoff_freq, filter_order)
