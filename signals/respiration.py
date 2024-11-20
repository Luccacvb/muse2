import numpy as np
from scipy.signal import detrend

def process_respiration(inlet, buffer):
    """Captura e processa os dados de respiração."""
    samples, _ = inlet.pull_chunk(timeout=1.0, max_samples=250)
    if samples:
        data = np.array(samples).T
        buffer = np.hstack((buffer[:, -250 + data.shape[1]:], data))  # Atualizar buffer

        # Extrair respiração do sinal
        respiration_signal = detrend(buffer[0])
        return buffer, respiration_signal
    return buffer, np.zeros_like(buffer[0])
