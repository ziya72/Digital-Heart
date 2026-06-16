from preprocessing.clean import clean_ecg
from preprocessing.normalize import normalize_ecg
from features.rpeak import detect_rpeaks
from features.rr import compute_rr
from features.hrv import extract_hrv_features

def process_signal(signal, fs):
    try:
        cleaned = clean_ecg(signal, fs)
        normalized = normalize_ecg(cleaned)
        rpeaks = detect_rpeaks(normalized, fs)

        if rpeaks is None or len(rpeaks) < 2:
            return None
        
        rr_intervals = compute_rr(rpeaks, fs)

        hrv_features = extract_hrv_features(rr_intervals)
        return hrv_features
    
    except Exception as e:
        print(f"Error processing signal: {e}")
        return None