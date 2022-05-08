import numpy as np

def leastsquares(x, y):
    """Given two vectors x and y of equal dimension, calculates
    the slope and y intercept of the y2 = c + m*x slope, obtained
    by least squares linear regression
    Documentation for numpy function used:
    https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.linalg.lstsq.html
    Output arguments
    c = y-intercept
    m = slope
    y2 = least square line"""

    # Rewriting the line equation as y = Ap, where A = [[x 1]]
    # and p = [[m], [c]]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=-1)[0]  # Finding coefficients m and c
    y2 = m*x+c  # Fitted line
    return m, c, y2