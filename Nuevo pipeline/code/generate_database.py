from parameters_calculation.clarity import clarity
from parameters_calculation.definition import definition
from parameters_calculation.tr_lundeby import tr_lundeby, NoiseError
from parameters_calculation.filtros import BandpassFilter
from librosa import load
import pandas as pd
import numpy as np
from progress.bar import IncrementalBar
from os import listdir

def generate_rir_descriptors(rirs_path, files, bands, filter_type, fs, order, max_ruido_dB):

    available_cache_files = listdir('cache')

    params_exists = False
    for available in available_cache_files:
        if f'descriptors_{max_ruido_dB}.pkl' in available:
            params_exists = True
            print('Parámetros calculados!')
    
    if params_exists:
        pass

    else:

        bpfilter = BandpassFilter(filter_type, fs, order, bands)
        
        bar = IncrementalBar('Calculating descriptors', max = len(files))

        name_df , band_df = [], []
        t30_df, c50_df, c80_df, d50_df = [], [], [], []

        for file in files:
            name = file.split('.wav')[0]
            data, _ = load(f'{rirs_path}/{file}', sr=fs) #Funciona
            filtered_audios = bpfilter.filtered_signals(data)

            for i, band in enumerate(bands):
                
                name_df.append(name)
                band_df.append(band)

                try:
                    t30, _, _ = tr_lundeby(filtered_audios[i], fs, max_ruido_dB)
                    c50 = clarity(50, filtered_audios[i], fs)
                    c80 = clarity(80, filtered_audios[i], fs)
                    d50 = definition(filtered_audios[i], fs)

                    t30_df.append(t30)
                    c50_df.append(c50)
                    c80_df.append(c80)
                    d50_df.append(d50)

                except (ValueError, NoiseError) as err:
                    print(err.args)
                    t30_df.append(np.nan)
                    c50_df.append(np.nan)
                    c80_df.append(np.nan)
                    d50_df.append(np.nan)
                    continue

            bar.next()

        bar.finish()
        
        data = {'RIR': name_df,
                'banda': band_df,
                't30': t30_df,
                'c50': c50_df,
                'c80': c80_df,
                'd50': d50_df}

        descriptors_df = pd.DataFrame(data)
        descriptors_df = descriptors_df.dropna() #Borro los nan así sé qué RIRs y bandas puedo usar
        
        pd.DataFrame(data).to_pickle(f'cache/descriptors_{max_ruido_dB}.pkl')

        print('Parámetros calculados!')

    

def generate_tae():
    pass