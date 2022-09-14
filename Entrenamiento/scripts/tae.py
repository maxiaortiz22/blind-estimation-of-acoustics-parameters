import numpy as np
from filtros import butter_bandpass_filter, butter_lowpass_filter, butter_highpass_filter
from librosa import resample
from scipy.signal import hilbert

#Función principal: TAE por frecuencia
def TAE(data, fs, octave_band):
  # Filtro la señal por banda de octava entre 125 y 4k Hz
  fs = fs #frecuencia de sampleo
  fc = octave_band #Frecuencia central

  if fc == 8000:
    cutoff = fc/np.sqrt(2) #Frecuencia de corte inferior
    order = 4 #Orden del filtro
    filtered_signal = butter_highpass_filter(data, cutoff, fs, order)
  else:
    lowcut = fc/np.sqrt(2) #Frecuencia de corte inferior
    highcut = fc*np.sqrt(2) #Frecuencia de corte  superior
    order = 4 #Orden del filtro
    filtered_signal = butter_bandpass_filter(data, lowcut, highcut, fs, order)

  #Transformada de Hilbert
  analytic_signal = hilbert(filtered_signal)
  #Obtengo la envolvente
  amplitude_envelope = np.abs(analytic_signal)

  #Obtengo la TAE a partir del resampleo y filtrado de la señal:
  cutoff = 20 # Frecuencia de corte a 20 Hz

  tae = butter_lowpass_filter(amplitude_envelope, cutoff, fs, order) # Le paso un filtro pasabajos a la envolvente

  #Downsampleo la señal:
  tae = resample(tae, orig_sr=fs, target_sr=40)

  tae = tae/np.max(np.abs(tae))

  #Si el audio tiene 5 [s] de duración y 16k de frec de sampleo, el tae debería ser de 200!
  assert(tae.size==200)

  return tae