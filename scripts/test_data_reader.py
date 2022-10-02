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
    random_state = 22
    bands = [125, 250, 500, 1000, 2000, 4000, 8000]
    max_ruido_dB = -60
    noise = False

    for band in bands:
        db = read_dataset(band, max_ruido_dB, noise, sample_frac, random_state)
        print(db)

    print("--- %s seconds ---" % (time.time() - start_time))