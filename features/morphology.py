import numpy as np
import neurokit2 as nk

def safe_interval(starts, ends, fs=500, lo_ms=50, hi_ms=600):
    s = np.array(starts, dtype=float)
    e = np.array(ends, dtype=float)
    diff = (e - s) / fs * 1000
    valid = (~np.isnan(diff)) & (diff > lo_ms) & (diff < hi_ms)
    return diff[valid]

def extract_morphology(signal_norm, signal_raw, fs=500):
    try:
        # ---- SELECT LEAD II ----
        # handle both 1D and 2D safely
        if signal_norm.ndim == 1:
            lead_norm = signal_norm
        else:
            lead_norm = signal_norm[:, 1]
        
        if signal_raw.ndim == 1:
            lead_raw = signal_raw
        else:
            lead_raw = signal_raw[:, 1]

        # ---- PROCESS ----
        if len(lead_norm) < 50 or np.all(np.isnan(lead_norm)):
            return None
        print("Lead length:", len(lead_norm))
        print("NaNs:", np.isnan(lead_norm).sum())
        
        signals, info = nk.ecg_process(lead_norm, sampling_rate=fs)
        _, waves = nk.ecg_delineate(
            lead_norm,
            info["ECG_R_Peaks"],
            sampling_rate=fs,
            method="dwt"
        )

        # ---- PULL KEYS ----
        p_onsets  = np.array(waves.get("ECG_P_Onsets",  []), dtype=float)
        q_peaks   = np.array(waves.get("ECG_Q_Peaks",   []), dtype=float)
        s_peaks   = np.array(waves.get("ECG_S_Peaks",   []), dtype=float)
        t_offsets = np.array(waves.get("ECG_T_Offsets", []), dtype=float)
        t_peaks   = np.array(waves.get("ECG_T_Peaks",   []), dtype=float)
        r_peaks   = np.array(info["ECG_R_Peaks"],            dtype=float)


        # ---- ALIGN LENGTH ----
        n = min(len(p_onsets), len(q_peaks), len(s_peaks), len(t_offsets), len(t_peaks))

        if n == 0:
            return None

        p_onsets  = np.array(p_onsets[:n], dtype=float)
        q_peaks   = np.array(q_peaks[:n], dtype=float)
        s_peaks   = np.array(s_peaks[:n], dtype=float)
        t_offsets = np.array(t_offsets[:n], dtype=float)
        t_peaks   = np.array(t_peaks[:n], dtype=float) if len(t_peaks) >= n else np.array([], dtype=float)

        # ---- INTERVALS ----
        pr_ms  = safe_interval(p_onsets, q_peaks,   fs, lo_ms=80, hi_ms=220)
        qrs_ms = safe_interval(q_peaks,  s_peaks,   fs, lo_ms=40, hi_ms=130)
        qt_ms  = safe_interval(q_peaks,  t_offsets, fs, lo_ms=250, hi_ms= 550)

        # ---- RR for QTc ----
        rr = np.diff(r_peaks[~np.isnan(r_peaks)]) / fs
        rr_mean = np.mean(rr) if len(rr) > 0 else 1.0

        qtc = qt_ms / np.sqrt(rr_mean) if len(qt_ms) > 0 else []

        # ---- AMPLITUDES ----
        r_idx = r_peaks[~np.isnan(r_peaks)].astype(int)
        r_idx = r_idx[r_idx < len(lead_norm)]
        r_amp = lead_norm[r_idx]
        
        t_idx = t_peaks[~np.isnan(t_peaks)].astype(int)
        t_idx = t_idx[t_idx < len(lead_norm)]
        t_amp = lead_norm[t_idx]


        # ---- ST DEVIATION ----

        # ---- ST DEVIATION from RAW lead (real mV) ----
        st_values = []
        for p, q in zip(p_onsets, q_peaks):
            # skip if either is NaN
            if np.isnan(p) or np.isnan(q):
                continue
            p, q = int(p), int(q)

            # baseline = mean of PR segment (10 samples before P onset)
            baseline_start = max(0, p - 10)
            baseline = np.mean(lead_raw[baseline_start:p]) if p > 0 else 0.0

            # ST point = 80ms after Q
            st_point = q + int(0.08 * fs)
            if st_point < len(lead_raw):
                st_values.append(float(lead_raw[st_point]) - baseline)
    

        return {
            "pr_mean": np.mean(pr_ms) if len(pr_ms) > 0 else None,
            "qrs_mean": np.mean(qrs_ms) if len(qrs_ms) > 0 else None,
            "qt_mean": np.mean(qt_ms) if len(qt_ms) > 0 else None,
            "qtc_mean": np.mean(qtc) if len(qtc) > 0 else None,
            "r_amp_mean": np.mean(r_amp) if len(r_amp) > 0 else None,
            "t_amp_mean": np.mean(t_amp) if len(t_amp) > 0 else None,
            "st_mean": np.mean(st_values) if len(st_values) > 0 else None
        }

    except Exception as e:
        print(f"Error in morphology: {e}")
        return None


