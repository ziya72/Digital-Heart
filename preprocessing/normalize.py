import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk

def normalize_ecg(signal):
    mean = np.mean(signal, axis=1, keepdims=True)
    std = np.std(signal, axis=1, keepdims=True)
    return (signal-mean)/(std + 1e-8)