import numpy as np

def center_time(signal, fs):
    """
    Calculate the Center time (Ts) descriptor from a given impulse response.
    """

    # Windows signal from its maximum onwards:
    #in_max = np.where(abs(signal) == np.max(abs(signal)))[0]  
    #in_max = int(in_max[0])
    #signal = signal[(in_max):]

    #Calculation of the descriptor:
    h2 = signal**2.0
    h3 = [(i/fs)*h2[i] for i, _ in enumerate(h2)]
    Ts = np.sum(h3) / np.sum(h2)
    return Ts*1000