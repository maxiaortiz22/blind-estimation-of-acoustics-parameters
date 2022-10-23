"""
Experimento 2: Entreno el modelo con toda la base de datos, sin agregar ruido rosa.

El criterio de aceptación de ruido para determinar si una RIR es válida o no será de 45 dB.

Para este caso uso los siguientes parámetros para la red:

    * filters = [32, 18, 8, 4] 
    * kernel_size = [10, 5, 5, 5] 
    * activation = ['relu','relu','relu','relu'] 
    * pool_size = [2,2,2]
    * learning_rate = 0.001
"""
import random
import os

#Configuración global:
seed = 2222 #Inicializador del generador de números random
exp_num = 2 #Número del experimento

# Data:
tot_rirs_from_data = len(os.listdir('data/RIRs')) #Cantidad de RIRs a agarrar de la carpeta data/RIRs
random.seed(seed) #Seed para hacer reproducible el random de agarrado de RIRs

#Parámetros para el cálculo de los descriptores:
files_rirs = os.listdir('data/RIRs') #Audios de las RIRs
files_rirs = random.sample(files_rirs, k=tot_rirs_from_data)
sinteticas_rirs = [audio for audio in files_rirs if 'sintetica' in audio]
tot_sinteticas = len(sinteticas_rirs)
files_speech = os.listdir('data/Speech') #Audios de voz
bands = [125, 250, 500, 1000, 2000, 4000, 8000] #Bandas a analizar
filter_type = 'octave band' #Tipo de filtro a utilizar: 'octave band' o 'third octave band'
fs = 16000 #Frecuencia de sampleo de los audios.
order = 4 #Orden del filtro
max_ruido_dB = -45 #Criterio de aceptación de ruido para determinar si una RIR es válida o no
add_noise = False #Booleano para definir si agregar ruido rosa o no a la base de datos.
snr = [-5, 20] #Valores de SNR que tendrían los audios si se les agrega ruido
tr_aug = [0.2, 3.1, 0.1] #Aumentar los valores de TR de 0.2 a 3 s con pasos de 0.1 s
drr_aug = [-6, 19, 1] #Aumentar los valores de DRR de -6 a 18 dB con pasos de 1 dB

#Parámetros para la lectura de la base de datos:
sample_frac = 1.0 #Fracción de la data a leer
random_state = seed #Inicializador del generador de números random

train = 0.8
test = 0.2

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