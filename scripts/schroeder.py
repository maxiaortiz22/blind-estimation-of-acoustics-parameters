import numpy as np

def schroeder(ir, t, C):
    """ Smooths a curve (ir) using Schroeder Integration method. "t" and "C" are Lundeby's compensation arguments """
    ir = ir[0:int(t)]
    y = np.flip((np.cumsum(np.flip(ir)) + C) / (np.sum(ir) + C))
    return y