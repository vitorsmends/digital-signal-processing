import numpy as np
import matplotlib.pyplot as plt

# Função para realizar a FFT e plotar o espectro de frequência
def plot_fft(vector, title, ax):
    fft_result = np.fft.fft(vector)
    freq = np.fft.fftfreq(len(vector))
    ax.stem(freq, np.abs(fft_result), use_line_collection=True)
    ax.set_title(title)
    ax.set_xlabel('Frequência')
    ax.set_ylabel('Magnitude')

# Função para ajustar vetores para terminarem com a mesma quantidade de zeros no final
def adjust_vector(vector):
    num_ones = np.sum(vector == 1)
    zeros_to_add = 2000 - num_ones
    adjusted_vector = np.concatenate((vector, np.zeros(zeros_to_add)))
    return adjusted_vector

# Criar os vetores ajustados
vector1 = adjust_vector(np.ones(10))
vector2 = adjust_vector(np.ones(100))
vector3 = adjust_vector(np.ones(1000))

# Criar subplots
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Plotar a FFT de cada vetor
plot_fft(vector1, 'FFT do vetor com tamanho 10', axs[0])
plot_fft(vector2, 'FFT do vetor com tamanho 100', axs[1])
plot_fft(vector3, 'FFT do vetor com tamanho 1000', axs[2])

plt.tight_layout()
plt.show()
