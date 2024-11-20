from scipy.signal import butter, filtfilt

def bandpass_filter(data, low, high, fs):
    """Filtro passa-banda para sinais."""
    nyquist = 0.5 * fs
    lowcut = low / nyquist
    highcut = high / nyquist
    b, a = butter(4, [lowcut, highcut], btype='band')
    return filtfilt(b, a, data, axis=1)
