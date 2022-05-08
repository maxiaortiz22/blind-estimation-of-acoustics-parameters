import numpy as np
from librosa import resample
from filtros import butter_bandpass_filter

def syntheticRIR_band(Rt, fs, fc):
  # Reconstruir una RIR con el método de schroeder a partir de conocer el tiempo de reverberación y
  # la banda de frecuencia a la cual pertenece.

  t = np.arange(0, int(Rt+1), 1/fs)

  a = 1 #Amplitud de la RIR
  M = 0.00001 #Factor de ruido de fondo

  y = a*np.e**((-6.9*t)/Rt) # Creo una exponencial decreciente
  
  #np.random.seed(2021)
  n = np.random.normal(0, 1, y.shape) #Genero ruido blanco

  fs = fs #frecuencia de sampleo
  fc = fc #Frecuencia central
  lowcut = fc/(6*np.sqrt(2)) #Frecuencia de corte inferior para tercio de octava
  highcut = fc*np.sqrt(2)*6 #Frecuencia de corte  superior para tercio de octava
  order = 4 #Orden del filtro


  filtered_signal = butter_bandpass_filter(n, lowcut, highcut, 44100, order)

  n = resample(filtered_signal, orig_sr=44100, target_sr=fs) #resampleo el ruido a la fs de la RIR

  z = y*n #Multiplico el ruido por la señal
  y = z + M*n #Agrego un poco de ruido de fondo a la señal

  return y/max(y)