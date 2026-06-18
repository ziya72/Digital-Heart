import numpy as np
import pandas as pd
import wfdb
from pipeline.process_signal import process_signal

def load_record(path):
    record = wfdb.rdsamp(path)
    signal = record[0]
    meta = record[1]
    return signal, meta


def main():
    df = pd.read_csv('data/ptb-xl/ptbxl_database.csv')

    all_features = []

    for i, row in df.iterrows():
        path = 'data/ptb-xl/' + row['filename_lr']

        signal, meta = load_record(path)
        fs = meta['fs']

        features = process_signal(signal, fs)

        if features is not None:
            all_features.append(features)

    
    print(f"Processed {len(all_features)} samples")

if __name__ == "__main__":
    main()