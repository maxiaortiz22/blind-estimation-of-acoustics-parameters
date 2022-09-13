import numpy as np
from random import randrange

def syntheticRIR(Rt, fs):
  # Reconstruir una RIR con el método de schroeder a partir de conocer el tiempo de reverberación

  t = np.arange(0, Rt+0.5, 1/fs)
  y = np.e**((-6.9*t)/Rt) # Creo una exponencial decreciente

  seed = randrange(int(2**32))
  
  np.random.seed(seed)
  n = np.random.normal(0, 1, y.shape) #Genero ruido blanco

  y = y*n #Multiplico el ruido por la señal

  return y/np.max(np.abs(y)), seed

if __name__ == '__main__':
    import soundfile as sf

    #Valores de TR entre 0.2 a 3 [s] con pasos de 0.1:
    TRs = np.arange(1.0, 1.51, 0.01, dtype=float)
    fs = 16000

    COUNT = 1
    for i, TR in enumerate(TRs):
        TR = np.round(TR, 2)

        for u in range(1, 201):
            RIR, seed = syntheticRIR(TR, fs) #Genero una RIR sintética
            sf.write(f'RIRs/sintetica_Seed{seed}_Tr{TR}.wav', RIR, fs) #Guardo el audio
            print(f'Se generaron {COUNT} RIRs sintéticas! sintetica_Seed{seed}_Tr{TR}.wav')
            COUNT+=1