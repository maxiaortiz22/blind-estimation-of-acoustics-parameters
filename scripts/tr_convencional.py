import numpy as np
from scipy import stats
import sys

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