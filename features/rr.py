#--------RR INTERVALS--------#

import numpy as np

def compute_rr(rpeaks, fs):

    if len(rpeaks) < 2:
        return None

    rr_intervals = np.diff(rpeaks) / fs

    return rr_intervals