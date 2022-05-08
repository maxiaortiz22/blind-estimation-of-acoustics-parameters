from scipy.signal import butter, sosfilt, lfilter, fftconvolve, hilbert
import numpy as np

#Filtros scipy:
def butter_bandpass(lowcut, highcut, fs, order):
    return butter(order, [lowcut, highcut], fs=fs, btype='bandpass', output='sos')

def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y

def butter_lowpass(cutoff, fs, order):
    return butter(order, cutoff, fs=fs, btype='lowpass', output='sos')

def butter_lowpass_filter(data, cutoff, fs, order):
    sos = butter_lowpass(cutoff, fs, order)
    y = sosfilt(sos, data)
    return y

def butter_highpass(cutoff, fs, order):
    return butter(order, cutoff, fs=fs, btype='highpass', output='sos')

def butter_highpass_filter(data, cutoff, fs, order):
    sos = butter_highpass(cutoff, fs, order)
    y = sosfilt(sos, data)
    return y

def bandpass_filtered_signals(data, fs, order, type='octave band'):
  # Voy a hacer un filtro pasa bandas para todos menos para el de 8k, en ese
  # caso voy a tener que usar un pasa altos porque sino las frecuencias de corte
  # me quedan fuera.
  bands = [125, 250, 500, 1000, 2000, 4000, 8000]

  filtered_audios = np.empty((len(bands), len(data))) # Array vacio para cargar las señales filtradas
  for i in range(int(len(bands)-1)): #Tomo de 125 a 4k
    fs = fs #frecuencia de sampleo
    fc = bands[i] #Frecuencia central
    order = order #Orden del filtro

    if type == 'octave band':
      lowcut = fc/np.sqrt(2) #Frecuencia de corte inferior bandas de octava
      highcut = fc*np.sqrt(2) #Frecuencia de corte  superior bandas de octava
    elif type == 'third octave band':
      lowcut = fc/(2**(1/6)) #Frecuencia de corte inferior bandas de tercio de octava
      highcut = fc*(2**(1/6)) #Frecuencia de corte  superior bandas de tercio de octava

    filtered_signal = butter_bandpass_filter(data, lowcut, highcut, fs, order) # Filtro la señal

    filtered_audios[i, :] = filtered_signal # Cargo la banda de 125 a 4k

  # Ahora calculo para 8k:
  if type == 'octave band':
    lowcut = fc/np.sqrt(2) #Frecuencia de corte inferior banda de octava
  elif type == 'third octave band':
    lowcut = fc/(2**(1/6)) #Frecuencia de corte inferior banda de tercio de octava
  
  filtered_audios[-1, :] = butter_highpass_filter(data, lowcut, fs, order) # Filtro la señal en 8k

  return filtered_audios