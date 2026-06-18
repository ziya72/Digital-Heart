from preprocessing.clean import clean_ecg
from preprocessing.normalize import normalize_ecg
from features.rpeak import detect_rpeaks
from features.rr import compute_rr
from features.hrv import extract_hrv_features
from features.morphology import extract_morphology

def process_signal(signal, fs):
    try:
        cleaned = clean_ecg(signal, fs)
        normalized = normalize_ecg(cleaned)
        rpeaks = detect_rpeaks(normalized, fs)
    
        if rpeaks is None or len(rpeaks) < 2:
            return None
        
        rr_intervals = compute_rr(rpeaks, fs)

        rr_intervals = rr_intervals[(rr_intervals > 0.3 ) & (rr_intervals < 2.0)]

        if len(rr_intervals) < 2:
            return None
        
        hrv_features = extract_hrv_features(rr_intervals)
        
        if hrv_features is None:
            return None
        
        morph_features = extract_morphology(normalized, fs)

        if morph_features is None:
            return None
        
        combined_features = {
            **hrv_features,
            **morph_features
        }

        return combined_features
    
    except Exception as e:
        print(f"Error processing signal: {e}")
        return None