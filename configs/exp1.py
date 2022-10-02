"""
Probando el funcionamiento del pipeline!
"""
from itertools import product
import os

#Configuración global:
seed = 2222 #Inicializador del generador de números random
exp_num = 1 #Número del experimento

# Data:

#Parámetros para el cálculo de los descriptores:
files_rirs = os.listdir('data/RIRs') #Audios de las RIRs
files_voices = os.listdir('data/Speech') #Audios de voz
bands = [125, 250, 500, 1000, 2000, 4000, 8000] #Bandas a analizar
filter_type = 'octave band' #Tipo de filtro a utilizar: 'octave band' o 'third octave band'
fs = 16000 #Frecuencia de sampleo de los audios.
order = 4 #Orden del filtro
max_ruido_dB = -60 #Criterio de aceptación de ruido para determinar si una RIR es válida o no
add_noise = False #Booleano para definir si agregar ruido rosa o no a la base de datos.
snr = [-5, 20] #Valores de SNR que tendrían los audios si se les agrega ruido

#Parámetros para la lectura de la base de datos:
sample_frac = 1.0 #Fracción de la data a leer
random_state = seed #Inicializador del generador de números random

# Modelo:

#Parámetros de la red:
filters = [32, 18, 8, 4] 
kernel_size = [10, 5, 5, 5] 
activation = ['relu','relu','relu','relu'] 
pool_size = [2,2,2]
learning_rate = 0.001

#Parámetros de entrenamiento:
validation_split = 0.1 
batch_size = 1024
epochs = 500