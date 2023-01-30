import os
os.chdir('../')

import pandas as pd
import random
import sys
sys.path.append('code')
from data_reader import read_dataset
#(rirs_path, files, bands, filter_type, fs, order, max_ruido_dB)

if __name__ == '__main__':
    import time
    start_time = time.time()

    tot_rirs_from_data = 10
    seed = 2222
    random.seed(seed)

    files_rirs = os.listdir('data/RIRs') #Audios de las RIRs
    files_rirs = random.sample(files_rirs, k=tot_rirs_from_data)
    sinteticas_rirs = [audio for audio in files_rirs if 'sintetica' in audio]
    tot_sinteticas = len(sinteticas_rirs)
    files_speech = os.listdir('data/Speech') #Audios de voz
    bands = [125, 250, 500, 1000, 2000, 4000, 8000] #Bandas a analizar
    filter_type = 'octave band' #Tipo de filtro a utilizar: 'octave band' o 'third octave band'
    fs = 16000 #Frecuencia de sampleo de los audios.
    order = 4 #Orden del filtro
    max_ruido_dB = -60 #Criterio de aceptación de ruido para determinar si una RIR es válida o no
    add_noise = False #Booleano para definir si agregar ruido rosa o no a la base de datos.
    snr = [-5, 20] #Valores de SNR que tendrían los audios si se les agrega ruido
    tr_aug = [0.2, 3.1, 0.1] #Aumentar los valores de TR de 0.2 a 3 s con pasos de 0.1 s
    drr_aug = [-6, 19, 1] #Aumentar los valores de DRR de -6 a 18 dB con pasos de 1 dB

    db_name = f'base_de_datos_{max_ruido_dB}_noise_{add_noise}_traug_{tr_aug[0]}_{tr_aug[1]}_{tr_aug[2]}_drraug_{drr_aug[0]}_{drr_aug[1]}_{drr_aug[2]}_snr_{snr[0]}_{snr[-1]}'

    for band in bands:
        db = read_dataset(band, db_name, 1.0, seed, 'train')
        print(len(db))
        print(db.head())

    print("--- %s seconds ---" % (time.time() - start_time))