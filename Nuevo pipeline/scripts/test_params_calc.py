import os
os.chdir('../')

import pandas as pd
import sys
sys.path.append('code')
from generate_database import generate_rir_descriptors, generate_tae
#(rirs_path, files, bands, filter_type, fs, order, max_ruido_dB)

if __name__ == '__main__':
    import time
    start_time = time.time()

    files_rirs = os.listdir('data/RIRs')
    files_voices = os.listdir('data/Speech')
    bands = [125, 250, 500, 1000, 2000, 4000, 8000]
    filter_type = 'octave band'
    fs = 16000
    order = 4
    max_ruido_dB = -45
    add_noise = False
    snr = [-5, 20] #Para sacar una SNR con np.random.uniform(snr[0], snr[-1], 1)[0]. Agregar esto al generate_database.py

    generate_rir_descriptors(files_rirs, bands, filter_type, fs, order, max_ruido_dB)

    #descriptors = pd.read_pickle(f'cache/descriptors_{max_ruido_dB}.pkl')

    #print(descriptors)
    #print(descriptors.to_string())

    generate_tae(files_voices, bands, filter_type, fs, max_ruido_dB, add_noise, snr)

    db = pd.read_pickle(f'cache/base_de_datos_{max_ruido_dB}.pkl')

    print(db)

    print("--- %s seconds ---" % (time.time() - start_time))