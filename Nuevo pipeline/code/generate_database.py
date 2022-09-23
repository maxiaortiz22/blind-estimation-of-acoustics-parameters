from parameters_calculation.clarity import clarity
from parameters_calculation.definition import definition
from parameters_calculation.tr_lundeby import tr_lundeby, NoiseError
from parameters_calculation.filtros import BandpassFilter
from parameters_calculation.tae import TAE
from librosa import load
import pandas as pd
from progress.bar import IncrementalBar
from os import listdir
from math import nan
from scipy.signal import butter, fftconvolve

def generate_rir_descriptors(files, bands, filter_type, fs, order, max_ruido_dB):

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
            data, _ = load(f'data/RIRs/{file}', sr=fs) #Funciona
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
                    t30_df.append(nan)
                    c50_df.append(nan)
                    c80_df.append(nan)
                    d50_df.append(nan)
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
        
        descriptors_df.to_pickle(f'cache/descriptors_{max_ruido_dB}.pkl')

        print('Parámetros calculados!')

    

def generate_tae(files, bands, filter_type, fs, max_ruido_dB, add_noise, snr):

    available_cache_files = listdir('cache')

    db_exists = False
    for available in available_cache_files:
        if f'base_de_datos_{max_ruido_dB}.pkl' in available:
            db_exists = True
            print('Base de datos calculada!')

    if db_exists:
        pass

    else:

        # Filtro pasabajos para la envolvente de la TAE (calculo una sola vez y lo paso como parámetro)
        cutoff = 20 # Frecuencia de corte a 20 Hz
        order = 4 # Orden del filtro
        sos_lowpass_filter = butter(order, cutoff, fs=fs, btype='lowpass', output='sos')

        bpfilter = BandpassFilter(filter_type, fs, order, bands)

        descriptors = pd.read_pickle(f'cache/descriptors_{max_ruido_dB}.pkl') #Leo los descriptores disponibles de cada RIR

        #RIRs_df_descriptors = descriptors.RIR.to_numpy()
        #bands_df_descriptors = descriptors.banda.to_numpy()
        #t30_df_descriptors = descriptors.t30.to_numpy()
        #c50_df_descriptors = descriptors.c50.to_numpy()
        #c80_df_descriptors = descriptors.c80.to_numpy()
        #d50_df_descriptors = descriptors.d50.to_numpy()

        name_df , band_df, tae_df = [], [], []
        t30_df, c50_df, c80_df, d50_df, snr_df = [], [], [], [], []

        previous_rir = None #Inicializo la rir que estoy viendo
        bar = IncrementalBar('Calculating data base', max = int(len(files)*len(descriptors.RIR.to_numpy())))

        for file in files:
            voice_data, _ = load(f'data/Speech/{file}', sr=fs, duration=5.0) #Audio de voz
            voice_name = file.split('.wav')[0]

            for i, rir in enumerate(descriptors.RIR.to_numpy()):

                if previous_rir != rir: #Chequeo que no la haya leído antes
                    rir_data, _ = load(f'data/RIRs/{rir}.wav', sr=fs) #Audio de rir
                    reverbed_audio = fftconvolve(voice_data, rir_data, mode='same') #Reverbero el audio
                    filtered_audios = bpfilter.filtered_signals(reverbed_audio) #Filtro la señal por bandas

                
                # FALTA AGREGAR LA FORMA DE AGREGAR RUIDO ROSA O NO!!!! SI add_noise == True
                
                idx_audio = bands.index(descriptors.banda.to_numpy()[i]) #Me devuelve el índice del audio que tengo que buscar dentro de los filtrados

                tae = TAE(filtered_audios[idx_audio], fs, sos_lowpass_filter) #Calculo el TAE

                name_df.append(f'{voice_name}_{rir}')
                band_df.append(descriptors.banda.to_numpy()[i])
                tae_df.append([tae]) #Lo agrego en un wrapper dentro de una lista para poder guardarlo en el df
                t30_df.append(descriptors.t30.to_numpy()[i])
                c50_df.append(descriptors.c50.to_numpy()[i])
                c80_df.append(descriptors.c80.to_numpy()[i])
                d50_df.append(descriptors.d50.to_numpy()[i])
                snr_df.append(nan) #CAMBIAR POR EL VALOR DE SNR SI add_noise == True

                previous_rir = rir #RIR revisada en esta iteración

                bar.next()

        bar.finish()
        
        data = {'ReverbedAudio': name_df,
                'banda': band_df,
                'tae': tae_df,
                't30': t30_df,
                'c50': c50_df,
                'c80': c80_df,
                'd50': d50_df,
                'snr': snr_df}
        
        db_df = pd.DataFrame(data)
        
        db_df.to_pickle(f'cache/base_de_datos_{max_ruido_dB}.pkl') # AGREGAR OTRA POR SI TENGO RUIDO O NO!!!!

        print('Base de datos calculada!')
