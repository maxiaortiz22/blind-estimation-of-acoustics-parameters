import glob
from librosa import load
from filtros import bandpass_filtered_signals
from tr_lundeby import tr_lundeby
from clarity import clarity
from definition import definition
import pandas as pd
import numpy as np

palabras = glob.glob('RIRs/*.wav')
bandas = [125, 250, 500, 1000, 2000, 4000, 8000]

names_df = []
banda_df = []
t30_list = []
c50_list = []
c80_list = []
d50_list = []
for palabra in palabras:
    data, fs = load(f'{palabra}', sr=48000)

    name = palabra.split('.wav')[0]
    name = name[5:]

    filtered_audios = bandpass_filtered_signals(data, fs, 4, type='third octave band')

    for i, band in enumerate(bandas):

        names_df.append(name)
        banda_df.append(band)

        t30 = tr_lundeby(filtered_audios[i], fs)[-1]
        c50 = clarity(50, filtered_audios[i], fs)
        c80 = clarity(80, filtered_audios[i], fs)
        d50 = definition(filtered_audios[i], fs)

        t30_list.append(np.round(t30, 3))
        c50_list.append(np.round(c80, 2))
        c80_list.append(np.round(c80, 2))
        d50_list.append(np.round(d50, 1))

    print(palabra)

df = pd.DataFrame(data= {'Audio':names_df,
                        'Banda':banda_df,
                        'T30 [s]':t30_list,
                        'C50 [dB]':c50_list,
                        'C80 [dB]':c80_list,
                        'D50 [%]':d50_list})

#df.to_csv('Parámetros python.csv', encoding = "utf-8",index=False)
df.to_excel('Parámetros python tercios.xlsx', encoding = "utf-8",index=False)