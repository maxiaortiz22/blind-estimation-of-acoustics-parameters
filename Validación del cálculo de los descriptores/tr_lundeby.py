import numpy as np
import sys
import math
from scipy import stats

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

def schroeder(ir, t, C):
    """ Smooths a curve (ir) using Schroeder Integration method. "t" and "C" are Lundeby's compensation arguments """
    ir = ir[0:int(t)]
    y = np.flip((np.cumsum(np.flip(ir)) + C) / (np.sum(ir) + C))
    return y

def tr_convencional(raw_signal, fs, rt='t30'):  # pylint: disable=too-many-locals
    """
    Reverberation time from an impulse response.
    :param file_name: name of the WAV file containing the impulse response.
    :param bands: Octave or third bands as NumPy array.
    :param rt: Reverberation time estimator. It accepts `'t30'`, `'t20'`, `'t10'` and `'edt'`.
    :returns: Reverberation time :math:`T_{60}`
    """

    rt = rt.lower()
    if rt == 't30':
        init = -5.0
        end = -35.0
        factor = 2.0
    elif rt == 't20':
        init = -5.0
        end = -25.0
        factor = 3.0
    elif rt == 't10':
        init = -5.0
        end = -15.0
        factor = 6.0
    elif rt == 'edt':
        init = 0.0
        end = -10.0
        factor = 6.0

    #Recorto la señal desde el máximo en adelante:
    in_max = np.where(abs(raw_signal) == np.max(abs(raw_signal)))[0]  # Windows signal from its maximum onwards.
    in_max = int(in_max[0])
    raw_signal = raw_signal[(in_max):]
    
    abs_signal = np.abs(raw_signal) / np.max(np.abs(raw_signal))

    # Schroeder integration
    sch = np.cumsum(abs_signal[::-1]**2)[::-1]
    sch_db = 10.0 * np.log10(sch / np.max(sch) + sys.float_info.epsilon)

    # Linear regression
    sch_init = sch_db[np.abs(sch_db - init).argmin()]
    sch_end = sch_db[np.abs(sch_db - end).argmin()]
    init_sample = np.where(sch_db == sch_init)[0][0]
    end_sample = np.where(sch_db == sch_end)[0][0]
    x = np.arange(init_sample, end_sample + 1) / fs
    y = sch_db[init_sample:end_sample + 1]
    slope, intercept = stats.linregress(x, y)[0:2]

    # Reverberation time (T30, T20, T10 or EDT)
    db_regress_init = (init - intercept) / slope
    db_regress_end = (end - intercept) / slope
    t60 = factor * (db_regress_end - db_regress_init)

    return t60

def lundeby(y, Fs, Ts):
    """Given IR response "y" and samplerate "Fs" function returns upper integration limit of
    Schroeder's integral. Window length in ms "Ts" indicates window sized of the initial averaging of the input signal,
    Luneby recommends this value to be in the 10 - 50 ms range."""

    y_power = y
    y_promedio = np.zeros(int(len(y) / Fs / Ts))
    eje_tiempo = np.zeros(int(len(y) / Fs / Ts))

    t = math.floor(len(y_power) / Fs / Ts)
    v = math.floor(len(y_power) / t)

    for i in range(0, t):
        y_promedio[i] = sum(y_power[i * v:(i + 1) * v]) / v
        eje_tiempo[i] = math.ceil(v / 2) + (i * v)

    # First estimate of the noise level determined from the energy present in the last 10% of input signal
    ruido_dB = 10 * np.log10(
        sum(y_power[round(0.9 * len(y_power)):len(y_power)]) / (0.1 * len(y_power)) / max(y_power))
    y_promediodB = 10 * np.log10(y_promedio / max(y_power) + sys.float_info.epsilon)

    # Decay slope is estimated from a linear regression between the time interval that contains the maximum of the
    # input signal (0 dB) and the first interval 10 dB above the initial noise level
    r = int(max(np.argwhere(y_promediodB > ruido_dB + 10)))
    m, c, rectacuadmin = leastsquares(eje_tiempo[0:r], y_promediodB[0:r])
    cruce = (ruido_dB - c) / m

    if ruido_dB > -20:  # Insufficient S/N ratio to perform Lundeby
        punto = len(y_power)
        C = None
    else:

        # Begin Luneby's iterations
        error = 1
        INTMAX = 25
        veces = 1
        while error > 0.0001 and veces <= INTMAX:

            # Calculates new time intervals for median, with aprox. p-steps per 10 dB
            p = 10  # Number of steps every 10 dB
            delta = abs(10 / m)  # Number of samples for the 10 dB decay slope
            v = math.floor(delta / p)  # Median calculation window
            if (cruce - delta) > len(y_power):
                t = math.floor(len(y_power) / v)
            else:
                t = math.floor(len(y_power[0:round(cruce - delta)]) / v)
            if t < 2:
                t = 2

            media = np.zeros(t)
            eje_tiempo = np.zeros(t)
            for i in range(0, t):
                media[i] = sum(y_power[i * v:(i + 1) * v]) / len(y_power[i * v:(i + 1) * v])
                eje_tiempo[i] = math.ceil(v / 2) + (i * v)
            mediadB = 10 * np.log10(media / max(y_power) + sys.float_info.epsilon)
            m, c, rectacuadmin = leastsquares(eje_tiempo, mediadB)

            # New median of the noise energy calculated, starting from the point of the decay line 10 dB under the cross-point
            noise = y_power[(round(abs(cruce + delta))):]
            if len(noise) < round(0.1 * len(y_power)):
                noise = y_power[round(0.9 * len(y_power)):]
            rms_dB = 10 * np.log10(sum(noise) / len(noise) / max(y_power) + sys.float_info.epsilon)

            # New cross-point
            error = abs(cruce - (rms_dB - c) / m) / cruce
            cruce = round((rms_dB - c) / m)
            veces += 1
    # output
    if cruce > len(y_power):
        punto = len(y_power)
    else:
        punto = cruce
    C = max(y_power) * 10 ** (c / 10) * math.exp(m / 10 / np.log10(math.exp(1)) * cruce) / (
                -m / 10 / np.log10(math.exp(1)))
    return punto, C

def tr_lundeby(y, fs):
    """TR calculates T20, T30 and EDT parameters given a smoothed energy response "y" and its samplerate "fs" """
    #Normalizo y obtengo el cuadrado de la señal
    
    y_fail = y

    y = y / max(y)
    y **= 2

    #Recorto la señal desde el máximo en adelante:
    in_max = np.where(abs(y) == np.max(abs(y)))[0]  # Windows signal from its maximum onwards.
    in_max = int(in_max[0])
    y = y[(in_max):]

    #try:

    #Encuentro los cortes de lundeby:
    t, C = lundeby(y, fs, 0.05)

    #Saco schroeder:
    sch = schroeder(y, t, C)
    sch = 10 * np.log10(sch / max(sch) + sys.float_info.epsilon)

    #Cálculo de los TRs:
    t = np.arange(0, len(sch) / fs, 1 / fs)

    i_max = np.where(sch == max(sch))                                   # Finds maximum of input vector
    sch = sch[int(i_max[0][0]):]
    i_edt = np.where((sch <= max(sch)) & (sch > (max(sch) - 10)))       # Index of values between 0 and -10 dB 
    i_10 = np.where((sch <= max(sch) - 5) & (sch > (max(sch) - 15)))    # Index of values between -5 and -25 dB
    i_20 = np.where((sch <= max(sch) - 5) & (sch > (max(sch) - 25)))    # Index of values between -5 and -25 dB
    i_30 = np.where((sch <= max(sch) - 5) & (sch > (max(sch) - 35)))    # Index of values between -5 and -35 dB

    t_edt = t[i_edt]
    t_10 = t[i_10]
    t_20 = t[i_20]
    t_30 = t[i_30]

    y_edt = sch[i_edt]
    y_t10 = sch[i_10]
    y_t20 = sch[i_20]
    y_t30 = sch[i_30]

    m_edt, c_edt, f_edt = leastsquares(t_edt, y_edt)  #leastsquares function used to find slope intercept and line of each parameter
    m_t10, c_t10, f_t10 = leastsquares(t_10, y_t10)
    m_t20, c_t20, f_t20 = leastsquares(t_20, y_t20)
    m_t30, c_t30, f_t30 = leastsquares(t_30, y_t30)

    EDT = -60 / m_edt                                 # EDT, T10, T20 and T30 calculations
    T10 = -60 / m_t10
    T20 = -60 / m_t20
    T30 = -60 / m_t30

    TRs = [EDT, T10, T20, T30]
    
    return TRs

