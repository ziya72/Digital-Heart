import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk

def clean_ecg(signal, fs):
    cleaned = []
    for i in range(signal.shape[1]):
        cleaned_lead = nk.ecg_clean(signal[:, i], sampling_rate=fs)
        cleaned.append(cleaned_lead)

    
    return np.array(cleaned)