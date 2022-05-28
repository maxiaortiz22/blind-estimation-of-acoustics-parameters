from lundeby import lundeby
import numpy as np
from schroeder import schroeder
import sys
from leastsquares import leastsquares
from tr_convencional import tr_convencional

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

    try:

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

    except Exception as err:

        print('No funcionó el Lundeby! Va por método convencional')

        t60 = tr_convencional(y_fail, fs, rt='t30')

        TRs = [t60, t60, t60, t60]

        return TRs