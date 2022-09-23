import os
os.chdir('../')

import pandas as pd
import sys
sys.path.append('code')
from generate_database import generate_rir_descriptors
#(rirs_path, files, bands, filter_type, fs, order, max_ruido_dB)

if __name__ == '__main__':
    rirs_path = 'data/RIRs'
    files = os.listdir('data/RIRs')
    bands = [125, 250, 500, 1000, 2000, 4000, 8000]
    filter_type = 'octave band'
    fs = 16000
    order = 4
    max_ruido_dB = -45

    generate_rir_descriptors(rirs_path, files, bands, filter_type, fs, order, max_ruido_dB)

    descriptors = pd.read_pickle(f'cache/descriptors_{max_ruido_dB}.pkl')

    #print(descriptors)
    print(descriptors.to_string())