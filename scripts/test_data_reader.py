import os
os.chdir('../')

import pandas as pd
import sys
sys.path.append('code')
from data_reader import read_dataset
#(rirs_path, files, bands, filter_type, fs, order, max_ruido_dB)

if __name__ == '__main__':
    import time
    start_time = time.time()

    sample_frac = 1.0
    bands = [125, 250, 500, 1000, 2000, 4000, 8000]
    random_state = 22

    max_ruido_dB = -60
    add_noise = False
    snr = [-5, 20]
    tr_aug = [0.2, 3.1, 0.1]
    drr_aug = [-6, 19, 1]

    db_name = f'base_de_datos_{max_ruido_dB}_noise_{add_noise}_traug_{tr_aug[0]}_{tr_aug[1]}_{tr_aug[2]}_drraug_{drr_aug[0]}_{drr_aug[1]}_{drr_aug[2]}_snr_{snr[0]}_{snr[-1]}.pkl'

    for band in bands:
        db = read_dataset(band, db_name, sample_frac, random_state)
        print(db)

    print("--- %s seconds ---" % (time.time() - start_time))