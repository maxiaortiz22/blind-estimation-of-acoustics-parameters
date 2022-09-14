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