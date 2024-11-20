import numpy as np
from scipy.signal import find_peaks

def process_ppg(inlet, buffer):
    """Captura e processa os sinais PPG."""
    samples, _ = inlet.pull_chunk(timeout=1.0, max_samples=250)
    if samples:
        data = np.array(samples).T  # Transpor para organizar por canais
        if buffer.shape[0] != data.shape[0]:  # Ajustar se o número de canais não for o esperado
            buffer = np.zeros((data.shape[0], 250))
        buffer = np.hstack((buffer[:, -250 + data.shape[1]:], data))  # Atualizar buffer

        print("PPG Data:", data)
        # Calcular batimentos cardíacos (BPM) apenas para o primeiro canal
        peaks, _ = find_peaks(buffer[0], distance=50)
        if len(peaks) > 1:
            bpm = len(peaks) * (60 / (buffer.shape[1] / 256))  # 256 é a taxa de amostragem
        else:
            bpm = 0
        return buffer, bpm
    print("PPG Data:", data)

    return buffer, 0
