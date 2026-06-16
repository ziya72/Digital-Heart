import neurokit2 as nk

def detect_rpeaks(signal, fs):
    lead = signal[1]   # Lead II
    _, info = nk.ecg_peaks(lead, sampling_rate=fs)
    return info["ECG_R_Peaks"]