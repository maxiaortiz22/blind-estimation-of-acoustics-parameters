from scipy.signal import butter, sosfilt
import numpy as np
#IMPLEMENTAR UNA FORMA DE GENERAR LOS FILTROS UNA SOLA VEZ PARA OPTIMIZAR LOS CÁLCULOS

class BandpassFilter:

    def __init__(self, type, fs, order, bands):
        self.type = type #str: 'octave band' or 'third octave band' 
        self.fs = fs # int: Frecuencia de sampleo
        self.order = order #int: Orden del filtro
        self.bands = bands #List: Lista de bandas a analizar, ej: [125, 250, 500, 1000, 2000, 4000, 8000]

        # Calculo los parámetros del filtro una única vez al inicializar la clase:
        self.sos = []

        for band in self.bands:
            # Hago esta separación porque trabajo con Fs de 16k en los audios! Por esto no puedo
            # encontrar los parámetros para el pasabanda centrado en 8k! Por esto, en este caso uso
            # un pasa altos!
            if band < 8000:
                if self.type == 'octave band':
                    lowcut = band/np.sqrt(2) #Frecuencia de corte inferior bandas de octava
                    highcut = band*np.sqrt(2) #Frecuencia de corte  superior bandas de octava
                elif self.type == 'third octave band':
                    lowcut = band/(2**(1/6)) #Frecuencia de corte inferior bandas de tercio de octava
                    highcut = band*(2**(1/6)) #Frecuencia de corte  superior bandas de tercio de octava

                self.sos.append(butter(self.order, [lowcut, highcut], fs=self.fs, btype='bandpass', output='sos'))
            
            elif band >= 8000:
                # Ahora calculo para 8k:
                if self.type == 'octave band':
                    cutoff = band/np.sqrt(2) #Frecuencia de corte inferior banda de octava
                elif self.type == 'third octave band':
                    cutoff = band/(2**(1/6)) #Frecuencia de corte inferior banda de tercio de octava
            
                self.sos.append(butter(self.order, cutoff, fs=self.fs, btype='highpass', output='sos'))


    def filtered_signals(self, data):

        filtered_audios = np.empty((len(self.bands), len(data))) # Array vacio para cargar las señales filtradas

        for i, sos in enumerate(self.sos):
            filtered_audios[i, :] = sosfilt(sos, data)

        return filtered_audios
            


#Filtros scipy pasabajos y pasaaltos:
def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    sos = butter(order, [lowcut, highcut], fs=fs, btype='bandpass', output='sos')
    y = sosfilt(sos, data)
    return y

def butter_lowpass_filter(data, cutoff, fs, order):
    sos = butter(order, cutoff, fs=fs, btype='lowpass', output='sos')
    y = sosfilt(sos, data)
    return y
