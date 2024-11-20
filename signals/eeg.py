from utils.filters import bandpass_filter
import numpy as np

def process_eeg(inlet, buffer):
    samples, _ = inlet.pull_chunk(timeout=1.0, max_samples=250)
    if samples:
        data = np.array(samples).T
        buffer = np.hstack((buffer[:, -250 + data.shape[1]:], data))
        filtered_buffer = bandpass_filter(buffer, low=1, high=50, fs=256)
        return filtered_buffer
    return buffer
