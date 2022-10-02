import numpy as np

def definition(signal, fs):
    """
    Calculate the Definition (D50) descriptor from a given impulse response.
    """

    # Windows signal from its maximum onwards:
    #in_max = np.where(abs(signal) == np.max(abs(signal)))[0]  
    #in_max = int(in_max[0])
    #signal = signal[(in_max):]

    #Calculation of the descriptor:
    time = 50 #50ms of integration
    h2 = signal**2.0
    t = int((time / 1000.0) * fs + 1)
    d = 100 * np.sum(h2[:t]) / np.sum(h2)
    return d