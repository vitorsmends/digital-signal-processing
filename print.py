import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment

def plot_waveform_and_fft(signal, sample_rate, title):
    plt.figure(figsize=(12, 6))
    
    # Plot waveform
    plt.subplot(2, 1, 1)
    time = np.arange(len(signal)) / sample_rate
    plt.plot(time, signal)
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    # Calculate FFT
    fft = np.fft.fft(signal)
    freq = np.fft.fftfreq(len(signal), 1/sample_rate)
    
    # Plot FFT
    plt.subplot(2, 1, 2)
    plt.stem(freq, np.abs(fft), use_line_collection=True)
    plt.title('FFT')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.xlim(0, sample_rate / 2)
    
    plt.tight_layout()
    plt.suptitle(title)
    plt.show()

def process_audio_file(input_file):
    audio = AudioSegment.from_mp3(input_file)
    sample_width = audio.sample_width
    sample_rate = audio.frame_rate
    channels = audio.channels
    signal = np.array(audio.get_array_of_samples())
    
    return signal, sample_rate

def main():
    input_file = "./input-data/input.mp3"
    output_file = "./output/output_lowpass.mp3"
    
    input_signal, sample_rate = process_audio_file(input_file)
    # Não está claro se você possui o arquivo de saída para plotar também
    
    plot_waveform_and_fft(input_signal, sample_rate, title='Original Audio')

if __name__ == "__main__":
    main()
