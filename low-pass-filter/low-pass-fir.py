from scipy.signal import firwin, lfilter
from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt

def apply_lowpass_filter(signal, cutoff_freq, sample_rate, filter_order=100):
    # Design the FIR filter
    nyquist_rate = sample_rate / 2
    normalized_cutoff_freq = cutoff_freq / nyquist_rate
    taps = firwin(filter_order, normalized_cutoff_freq, window='hamming')
    
    # Apply the filter to the signal
    filtered_signal = lfilter(taps, 1.0, signal)
    return filtered_signal.astype(np.int16), taps

def plot_signal_and_fft(signal, sample_rate, title):
    # Plot temporal signal
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(len(signal)) / sample_rate, signal)
    plt.title(title + ' - Temporal Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # Calculate FFT
    fft_signal = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), d=1/sample_rate)
    
    # Plot FFT
    plt.subplot(2, 1, 2)
    plt.plot(freqs[:len(freqs)//2], np.abs(fft_signal)[:len(freqs)//2])
    plt.title(title + ' - FFT')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.tight_layout()
    plt.show()

def apply_lowpass_filter_to_audio_file(input_file, output_file, cutoff_freq, filter_order=100):
    # Load the audio file
    audio = AudioSegment.from_file(input_file)
    
    # Extract audio data and properties
    sample_width = audio.sample_width
    sample_rate = audio.frame_rate
    print("sample rate: {}".format(audio.frame_rate))
    channels = audio.channels
    
    # Convert audio to numpy array
    signal = np.array(audio.get_array_of_samples())
    
    # Plot input signal and FFT
    plot_signal_and_fft(signal, sample_rate, "Input")
    
    # Apply low-pass filter
    filtered_signal, taps = apply_lowpass_filter(signal, cutoff_freq, sample_rate, filter_order)
    
    # Plot filtered signal and FFT
    plot_signal_and_fft(filtered_signal, sample_rate, "Output")
    
    # Convert numpy array back to audio
    filtered_audio = AudioSegment(filtered_signal.tobytes(), 
                                  frame_rate=sample_rate,
                                  sample_width=sample_width,
                                  channels=channels)
    
    # Save filtered audio to file
    filtered_audio.export(output_file, format="mp3")

# Example usage
input_file = "./input-data/input.mp3"
output_file = "./output-data/output_lowpass.mp3"
cutoff_freq = 1000  # Adjust cutoff frequency as needed
filter_order = 1000  # Adjust filter order as needed

apply_lowpass_filter_to_audio_file(input_file, output_file, cutoff_freq, filter_order)
