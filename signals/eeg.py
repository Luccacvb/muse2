from scipy.signal import butter, filtfilt
from scipy.fftpack import fft
import numpy as np
from utils.filters import bandpass_filter

def process_eeg(inlet, buffer):
    samples, _ = inlet.pull_chunk(timeout=1.0, max_samples=250)
    if not samples:
        print("Nenhum dado recebido. Muse desconectado? Zerar buffer...")
        buffer.fill(0)
        return buffer

    # Atualizar buffer com novos dados
    data = np.array(samples).T
    buffer = np.hstack((buffer[:, -250 + data.shape[1]:], data))  # Últimas 250 amostras
    filtered_buffer = bandpass_filter(buffer, low=1, high=50, fs=256)

    # Cálculo das bandas de frequência
    band_powers = calculate_band_powers(filtered_buffer, fs=256)
    print("Potências das bandas:", band_powers)

    return filtered_buffer

def calculate_band_powers(eeg_data, fs=256):
    """
    Calcula a potência das bandas de frequência (Delta, Theta, Alpha, Beta, Gamma).
    """
    fft_values = np.abs(fft(eeg_data, axis=1)) / eeg_data.shape[1]  # Normalização proporcional
    freqs = np.fft.fftfreq(eeg_data.shape[1], 1 / fs)

    # Apenas frequências positivas
    fft_values = fft_values[:, freqs > 0]
    freqs = freqs[freqs > 0]

    # Definir bandas de frequência
    bands = {
        "Delta": (0.5, 4),
        "Theta": (4, 8),
        "Alpha": (8, 13),
        "Beta": (13, 30),
        "Gamma": (30, 50),
    }

    band_powers = {}
    for band, (low, high) in bands.items():
        idx = np.where((freqs >= low) & (freqs <= high))[0]
        band_powers[band] = np.mean(fft_values[:, idx], axis=1)  # Média direta sem log

    return band_powers


