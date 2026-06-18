# features/hrv.py

import numpy as np

def extract_hrv_features(rr):

    if rr is None or len(rr) < 2:
        return None

    rr_diff = np.diff(rr)

    mean_rr = np.mean(rr)
    std_rr = np.std(rr)
    min_rr = np.min(rr)
    max_rr = np.max(rr)


    rmssd = np.sqrt(np.mean(rr_diff ** 2))

    nn50 = np.sum(np.abs(rr_diff) > 0.05)
    pnn50 = nn50 / len(rr_diff)
    heart_rate = 60/mean_rr

    return {
        "mean_rr": float(mean_rr*1000),
        "std_rr": float(std_rr*1000),
        "min_rr": float(min_rr*1000),
        "max_rr": float(max_rr*1000),
        "rmssd": float(rmssd*1000),
        "nn50": int(nn50),
        "pnn50": float(pnn50),
        "heart_rate": float(heart_rate)
    }