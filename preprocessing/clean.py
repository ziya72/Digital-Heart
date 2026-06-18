import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk

def clean_ecg(signal, fs):
    """Accepts (5000,12) or (12,5000), always returns (12, 5000)."""
    # ensure leads-first shape
    if signal.shape[0] > signal.shape[1]:   # (5000, 12) → transpose
        signal = signal.T                    # now (12, 5000)
    
    cleaned = []
    for i in range(signal.shape[0]):        # iterate over 12 leads
        cleaned_lead = nk.ecg_clean(signal[i], sampling_rate=fs)
        cleaned.append(cleaned_lead)
    
    return np.array(cleaned)                # (12, 5000)
