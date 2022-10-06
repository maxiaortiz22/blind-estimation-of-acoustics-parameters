import os
os.chdir('../')

import pandas as pd
import sys
sys.path.append('code')
from generate_database import calc_database
#(rirs_path, files, bands, filter_type, fs, order, max_ruido_dB)

if __name__ == '__main__':
    import time
    start_time = time.time()

    files_rirs = os.listdir('data/RIRs')
    files_speech = os.listdir('data/Speech')
    bands = [125, 250, 500, 1000, 2000, 4000, 8000]
    filter_type = 'octave band'
    fs = 16000
    order = 4
    max_ruido_dB = -60
    add_noise = False
    snr = [-5, 20] #Para sacar una SNR con np.random.uniform(snr[0], snr[-1], 1)[0]. Agregar esto al generate_database.py
    tr_aug = [0.2, 3.1, 0.1]
    drr_aug = [-6, 19, 1]
    
    db_name = calc_database(files_speech, files_rirs, bands, filter_type, fs, max_ruido_dB, 
                            order, add_noise, snr, tr_aug, drr_aug)
 
    db = pd.read_pickle(f'cache/{db_name}')
    
    print(db)

    print("--- %s seconds ---" % (time.time() - start_time))