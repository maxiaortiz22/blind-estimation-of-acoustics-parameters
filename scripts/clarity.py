import numpy as np
from lundeby import lundeby
from schroeder import schroeder

def clarity(time, signal, fs):
    """
    Calculate the Clarity descriptor from a given impulse response.

    The 'time' variable could bo 50 (C50) or 80 (C80).
    """

    #Calculation of the descriptor:
    h2 = signal**2.0
    t = int((time / 1000.0) * fs + 1) #Así venía el original
    #t = int(time / 1000.0) * fs 
    c = 10.0 * np.log10((np.sum(h2[:t]) / np.sum(h2[t:])))
    return c