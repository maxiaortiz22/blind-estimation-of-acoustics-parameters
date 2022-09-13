import numpy as np

def rms(signal): 
    return np.sqrt(np.mean(signal**2))

def snr(signal_rms, noise_rms): 
    return 10*np.log10((signal_rms**2) / (noise_rms**2))

def rms_comp(signal_rms, noise_rms, snr_required):

    rms_required = np.sqrt((signal_rms**2)/(10**(snr_required/10)))
    return rms_required / noise_rms