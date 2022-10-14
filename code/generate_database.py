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
from parameters_calculation.pink_noise import pink_noise
import numpy as np
import parameters_calculation.snr_calculator as snr_calculator
from parameters_calculation.tr_augmentation import tr_augmentation, TrAugmentationError
from parameters_calculation.drr_augmentation import drr_aug
import os

def calc_database(speech_files, rir_files, tot_sinteticas, bands, filter_type, fs, max_ruido_dB, order, add_noise, snr, TR_aug, DRR_aug):
    
    available_cache_files = listdir('cache')

    db_name = f'base_de_datos_{max_ruido_dB}_noise_{add_noise}_traug_{TR_aug[0]}_{TR_aug[1]}_{TR_aug[2]}_drraug_{DRR_aug[0]}_{DRR_aug[1]}_{DRR_aug[2]}_snr_{snr[0]}_{snr[-1]}.gzip'

    db_exists = False
    for available in available_cache_files:
        if db_name in available:
            db_exists = True
            print('Base de datos calculada!')
    
    if db_exists:
        return db_name

    else: #Calculo la base de dato si no existe

        # Filtro pasabajos para la envolvente de la TAE (calculo una sola vez y lo paso como parámetro)
        cutoff = 20 # Frecuencia de corte a 20 Hz
        sos_lowpass_filter = butter(order, cutoff, fs=fs, btype='lowpass', output='sos')

        TR_variations = np.arange(TR_aug[0], TR_aug[1], TR_aug[2]) #De tr_aug[0] a tr_aug[1] s con pasos de tr_aug[2] s
        DRR_variations = np.arange(DRR_aug[0], DRR_aug[1], DRR_aug[2]) #De drr_aug[0] a drr_aug[1] dB con pasos de drr_aug[2] dB

        bpfilter = BandpassFilter(filter_type, fs, order, bands) #Instancio la clase de los filtros
        
        descriptors_df, name_df, band_df = [], [], []
        tae_df, snr_df = [], []

        tot_audios = int(len(speech_files)*len(rir_files) + 
                         len(speech_files)*(len(rir_files)-tot_sinteticas)*len(TR_variations) + 
                         len(speech_files)*(len(rir_files)-tot_sinteticas)*len(DRR_variations))

        bar = IncrementalBar('Calculating data base', max = tot_audios)
        
        for speech_file in speech_files: #Por cada audio de voz

            speech_name = speech_file.split('.wav')[0]
            speech_data, _ = load(f'data/Speech/{speech_file}', sr=fs, duration=5.0)
            speech_data = speech_data/np.max(np.abs(speech_data))
            
            for rir_file in rir_files: #Por cada RIR

                #Obtengo la rir
                rir_name = rir_file.split('.wav')[0]
                rir_data, _ = load(f'data/RIRs/{rir_file}', sr=fs)
                rir_data = rir_data/np.max(np.abs(rir_data))

                if 'sintetica' in rir_name: #Si se trata de una RIR sintética, realizo el cálculo sin aumentar

                    #Realizo el cálculo para la RIR original:

                    #Reverbero el audio:
                    reverbed_audio = fftconvolve(speech_data, rir_data, mode='same') #Reverbero el audio
                    reverbed_audio = reverbed_audio/np.max(np.abs(reverbed_audio))

                    filtered_speech = bpfilter.filtered_signals(reverbed_audio) #Filtro la señal de voz por bandas

                    filtered_rir = bpfilter.filtered_signals(rir_data) #Filtro la rir por bandas

                    name = f'{speech_name}|{rir_name}|original' #Nombre del archivo en la base de datos

                    for i, band in enumerate(bands):

                        try:
                            #Calculo los descriptores:
                            t30, _, _ = tr_lundeby(filtered_rir[i], fs, max_ruido_dB)
                            c50 = clarity(50, filtered_rir[i], fs)
                            c80 = clarity(80, filtered_rir[i], fs)
                            d50 = definition(filtered_rir[i], fs)

                            #Cálculo del tae:
                            if add_noise:
                                #Genero ruido rosa:
                                noise_data = pink_noise(len(filtered_speech[i]))

                                #Agrego ruido para tener SNR entre snr[0] y snr[-1] dB:
                                rms_signal = snr_calculator.rms(filtered_speech[i])
                                rms_noise = snr_calculator.rms(noise_data)

                                snr_required = np.random.uniform(snr[0], snr[-1], 1)[0]

                                comp = snr_calculator.rms_comp(rms_signal, rms_noise, snr_required)

                                noise_data_comp = noise_data*comp

                                reverbed_noisy_audio = reverbed_audio + noise_data_comp
                                reverbed_noisy_audio = reverbed_noisy_audio / np.max(np.abs(reverbed_noisy_audio))

                                tae = TAE(reverbed_noisy_audio, fs, sos_lowpass_filter) #Calculo el TAE
                                snr_df.append(snr_required)
                            
                            elif add_noise == False:
                                tae = TAE(filtered_speech[i], fs, sos_lowpass_filter) #Calculo el TAE
                                snr_df.append(nan) 
                            

                            #Guardo los valores:
                            descriptors_df.append([t30, c50, c80, d50])
                            name_df.append(name)
                            band_df.append(band)
                            tae_df.append(list(tae))
                            
                        
                        except (ValueError, NoiseError, Exception) as err:
                            #Paso a otra banda si hubo algún error en el cálculo de los descriptores:
                            print(err.args)
                            continue

                    bar.next()

                else: #Si no es sintética, realizo el cálculo en la original y la aumento:

                    #Realizo el cálculo para la RIR original:

                    #Reverbero el audio:
                    reverbed_audio = fftconvolve(speech_data, rir_data, mode='same') #Reverbero el audio
                    reverbed_audio = reverbed_audio/np.max(np.abs(reverbed_audio))

                    filtered_speech = bpfilter.filtered_signals(reverbed_audio) #Filtro la señal de voz por bandas

                    filtered_rir = bpfilter.filtered_signals(rir_data) #Filtro la rir por bandas

                    name = f'{speech_name}|{rir_name}|original' #Nombre del archivo en la base de datos

                    for i, band in enumerate(bands):

                        try:
                            #Calculo los descriptores:
                            t30, _, _ = tr_lundeby(filtered_rir[i], fs, max_ruido_dB)
                            c50 = clarity(50, filtered_rir[i], fs)
                            c80 = clarity(80, filtered_rir[i], fs)
                            d50 = definition(filtered_rir[i], fs)

                            #Cálculo del tae:
                            if add_noise:
                                #Genero ruido rosa:
                                noise_data = pink_noise(len(filtered_speech[i]))

                                #Agrego ruido para tener SNR entre snr[0] y snr[-1] dB:
                                rms_signal = snr_calculator.rms(filtered_speech[i])
                                rms_noise = snr_calculator.rms(noise_data)

                                snr_required = np.random.uniform(snr[0], snr[-1], 1)[0]

                                comp = snr_calculator.rms_comp(rms_signal, rms_noise, snr_required)

                                noise_data_comp = noise_data*comp

                                reverbed_noisy_audio = reverbed_audio + noise_data_comp
                                reverbed_noisy_audio = reverbed_noisy_audio / np.max(np.abs(reverbed_noisy_audio))

                                tae = TAE(reverbed_noisy_audio, fs, sos_lowpass_filter) #Calculo el TAE
                                snr_df.append(snr_required)
                            
                            elif add_noise == False:
                                tae = TAE(filtered_speech[i], fs, sos_lowpass_filter) #Calculo el TAE
                                snr_df.append(nan) 
                            

                            #Guardo los valores:
                            descriptors_df.append([t30, c50, c80, d50])
                            name_df.append(name)
                            band_df.append(band)
                            tae_df.append(list(tae))
                            
                        
                        except (ValueError, NoiseError, Exception) as err:
                            #Paso a otra banda si hubo algún error en el cálculo de los descriptores:
                            print(err.args)
                            continue

                    bar.next()

                    #Realizo el cálculo para las RIR aumentadas:

                    for TR_var in TR_variations: #Por cada aumentación de TR
                        try:
                            #Realizo la aumentación por TR:
                            rir_aug = tr_augmentation(rir_data, fs, TR_var, bpfilter)
                            rir_aug = rir_aug/np.max(np.abs(rir_aug))

                            #Reverbero el audio en el caso que sea posible aumentar:
                            reverbed_audio = fftconvolve(speech_data, rir_aug, mode='same') #Reverbero el audio
                            reverbed_audio = reverbed_audio/np.max(np.abs(reverbed_audio))  

                            filtered_speech = bpfilter.filtered_signals(reverbed_audio) #Filtro la señal de voz por bandas

                            filtered_rir = bpfilter.filtered_signals(rir_aug) #Filtro la rir por bandas

                            name = f'{speech_name}|{rir_name}|TR_var_{TR_var}' #Nombre del archivo en la base de datos
                            
                            for i, band in enumerate(bands):

                                try:
                                    #Calculo los descriptores:
                                    t30, _, _ = tr_lundeby(filtered_rir[i], fs, max_ruido_dB)
                                    c50 = clarity(50, filtered_rir[i], fs)
                                    c80 = clarity(80, filtered_rir[i], fs)
                                    d50 = definition(filtered_rir[i], fs)

                                    #Cálculo del tae:

                                    if add_noise:
                                        #Genero ruido rosa:
                                        noise_data = pink_noise(len(filtered_speech[i]))

                                        #Agrego ruido para tener SNR entre snr[0] y snr[-1] dB:
                                        rms_signal = snr_calculator.rms(filtered_speech[i])
                                        rms_noise = snr_calculator.rms(noise_data)

                                        snr_required = np.random.uniform(snr[0], snr[-1], 1)[0]

                                        comp = snr_calculator.rms_comp(rms_signal, rms_noise, snr_required)

                                        noise_data_comp = noise_data*comp

                                        reverbed_noisy_audio = reverbed_audio + noise_data_comp
                                        reverbed_noisy_audio = reverbed_noisy_audio / np.max(np.abs(reverbed_noisy_audio))

                                        tae = TAE(reverbed_noisy_audio, fs, sos_lowpass_filter) #Calculo el TAE
                                        snr_df.append(snr_required)
                                    
                                    elif add_noise == False:
                                        tae = TAE(filtered_speech[i], fs, sos_lowpass_filter) #Calculo el TAE
                                        snr_df.append(nan) 
                                    

                                    #Guardo los valores:
                                    descriptors_df.append([t30, c50, c80, d50])
                                    name_df.append(name)
                                    band_df.append(band)
                                    tae_df.append(list(tae))

                                    
                                
                                except (ValueError, NoiseError, Exception) as err:
                                    #Paso a otra banda si hubo algún error en el cálculo de los descriptores:
                                    print(err.args)
                                    continue
                            
                            bar.next()

                        except TrAugmentationError as err:
                            print(err.args)
                            bar.next()
                            continue

                    for DRR_var in DRR_variations: #Por cada aumentación de DRR
                        try:
                            #Realizo la aumentación por TR:
                            rir_aug = drr_aug(rir_data, fs, DRR_var)
                            rir_aug = rir_aug/np.max(np.abs(rir_aug))

                            #Reverbero el audio en el caso que sea posible aumentar:
                            reverbed_audio = fftconvolve(speech_data, rir_aug, mode='same') #Reverbero el audio
                            reverbed_audio = reverbed_audio/np.max(np.abs(reverbed_audio))  

                            filtered_speech = bpfilter.filtered_signals(reverbed_audio) #Filtro la señal de voz por bandas

                            filtered_rir = bpfilter.filtered_signals(rir_aug) #Filtro la rir por bandas

                            name = f'{speech_name}|{rir_name}|DRR_var_{DRR_var}' #Nombre del archivo en la base de datos
                            
                            for i, band in enumerate(bands):

                                try:
                                    #Calculo los descriptores:
                                    t30, _, _ = tr_lundeby(filtered_rir[i], fs, max_ruido_dB)
                                    c50 = clarity(50, filtered_rir[i], fs)
                                    c80 = clarity(80, filtered_rir[i], fs)
                                    d50 = definition(filtered_rir[i], fs)

                                    #Calculo del tae:

                                    if add_noise:
                                        #Genero ruido rosa:
                                        noise_data = pink_noise(len(filtered_speech[i]))

                                        #Agrego ruido para tener SNR entre snr[0] y snr[-1] dB:
                                        rms_signal = snr_calculator.rms(filtered_speech[i])
                                        rms_noise = snr_calculator.rms(noise_data)

                                        snr_required = np.random.uniform(snr[0], snr[-1], 1)[0]

                                        comp = snr_calculator.rms_comp(rms_signal, rms_noise, snr_required)

                                        noise_data_comp = noise_data*comp

                                        reverbed_noisy_audio = reverbed_audio + noise_data_comp
                                        reverbed_noisy_audio = reverbed_noisy_audio / np.max(np.abs(reverbed_noisy_audio))

                                        tae = TAE(reverbed_noisy_audio, fs, sos_lowpass_filter) #Calculo el TAE
                                        snr_df.append(snr_required)
                                    
                                    elif add_noise == False:
                                        tae = TAE(filtered_speech[i], fs, sos_lowpass_filter) #Calculo el TAE
                                        snr_df.append(nan) 
                                    

                                    #Guardo los valores:
                                    descriptors_df.append([t30, c50, c80, d50])
                                    name_df.append(name)
                                    band_df.append(band)
                                    tae_df.append(list(tae))


                                except (ValueError, NoiseError, Exception) as err:
                                    #Paso a otra banda si hubo algún error en el cálculo de los descriptores:
                                    print(err.args)
                                    continue

                            bar.next()

                        except Exception as err:
                            print(err.args)
                            bar.next()
                            continue

        bar.finish()

        #Genero el dataframe:
        data = {'ReverbedAudio': name_df,
                'banda': band_df,
                'tae': tae_df,
                'descriptors': descriptors_df,
                'snr': snr_df}
        
        db_df = pd.DataFrame(data)

        #db_df.to_pickle(f'cache/{db_name}')
        db_df.to_parquet(f'cache/{db_name}', compression='gzip')

        print('Base de datos calculada!')

        return db_name

class DataBase():
    """Clase para generar el cálculo de la base de datos para entrenar la red."""

    def __init__(self, speech_files, rir_files, tot_sinteticas, bands, filter_type, fs, max_ruido_dB, order, add_noise, snr, TR_aug, DRR_aug):
        self.speech_files = speech_files
        self.rir_files = rir_files
        self.tot_sinteticas = tot_sinteticas
        self.bands = bands
        self.filter_type = filter_type
        self.fs = fs
        self.max_ruido_dB = max_ruido_dB
        self.order = order
        self.add_noise = add_noise
        self.snr = snr
        self.TR_aug = TR_aug
        self.DRR_aug = DRR_aug

        self.db_name = f'base_de_datos_{max_ruido_dB}_noise_{add_noise}_traug_{TR_aug[0]}_{TR_aug[1]}_{TR_aug[2]}_drraug_{DRR_aug[0]}_{DRR_aug[1]}_{DRR_aug[2]}_snr_{snr[0]}_{snr[-1]}'

        # Filtro pasabajos para la envolvente de la TAE (calculo una sola vez y lo paso como parámetro)
        self.cutoff = 20 # Frecuencia de corte a 20 Hz
        self.sos_lowpass_filter = butter(self.order, self.cutoff, fs=self.fs, btype='lowpass', output='sos')

        self.TR_variations = np.arange(self.TR_aug[0], self.TR_aug[1], self.TR_aug[2]) #De tr_aug[0] a tr_aug[1] s con pasos de tr_aug[2] s
        self.DRR_variations = np.arange(self.DRR_aug[0], self.DRR_aug[1], self.DRR_aug[2]) #De drr_aug[0] a drr_aug[1] dB con pasos de drr_aug[2] dB

        self.bpfilter = BandpassFilter(self.filter_type, self.fs, self.order, self.bands) #Instancio la clase de los filtros

        self.tot_audios = int(len(rir_files))

        self.i = 0

        #self.tot_audios = int(len(self.speech_files)*len(self.rir_files) + 
        #                      len(self.speech_files)*(len(self.rir_files)-self.tot_sinteticas)*len(self.TR_variations) + 
        #                      len(self.speech_files)*(len(self.rir_files)-self.tot_sinteticas)*len(self.DRR_variations))

    def calc_database_multiprocess(self, rir_file):
        """Calculo de la base de datos utilizando técnicas de multiprocessing"""
        available_cache_files = listdir('cache')

        db_exists = False
        for available in available_cache_files:
            if self.db_name in available:
                db_exists = True
                print('Base de datos calculada!')
        
        if db_exists:
            pass

        else: #Calculo la base de dato si no existe

            descriptors_df, name_df, band_df = [], [], []
            tae_df, snr_df = [], []
            
            for speech_file in self.speech_files: #Por cada audio de voz

                speech_name = speech_file.split('.wav')[0]
                speech_data, _ = load(f'data/Speech/{speech_file}', sr=self.fs, duration=5.0)
                speech_data = speech_data/np.max(np.abs(speech_data))
                

                #Obtengo la rir
                rir_name = rir_file.split('.wav')[0]
                rir_data, _ = load(f'data/RIRs/{rir_file}', sr=self.fs)
                rir_data = rir_data/np.max(np.abs(rir_data))

                if 'sintetica' in rir_name: #Si se trata de una RIR sintética, realizo el cálculo sin aumentar

                    #Realizo el cálculo para la RIR original:

                    #Reverbero el audio:
                    reverbed_audio = fftconvolve(speech_data, rir_data, mode='same') #Reverbero el audio
                    reverbed_audio = reverbed_audio/np.max(np.abs(reverbed_audio))

                    filtered_speech = self.bpfilter.filtered_signals(reverbed_audio) #Filtro la señal de voz por bandas

                    filtered_rir = self.bpfilter.filtered_signals(rir_data) #Filtro la rir por bandas

                    name = f'{speech_name}|{rir_name}|original' #Nombre del archivo en la base de datos

                    for i, band in enumerate(self.bands):

                        try:
                            #Calculo los descriptores:
                            t30, _, _ = tr_lundeby(filtered_rir[i], self.fs, self.max_ruido_dB)
                            c50 = clarity(50, filtered_rir[i], self.fs)
                            c80 = clarity(80, filtered_rir[i], self.fs)
                            d50 = definition(filtered_rir[i], self.fs)

                            #Cálculo del tae:
                            if self.add_noise:
                                #Genero ruido rosa:
                                noise_data = pink_noise(len(filtered_speech[i]))

                                #Agrego ruido para tener SNR entre snr[0] y snr[-1] dB:
                                rms_signal = snr_calculator.rms(filtered_speech[i])
                                rms_noise = snr_calculator.rms(noise_data)

                                snr_required = np.random.uniform(self.snr[0], self.snr[-1], 1)[0]

                                comp = snr_calculator.rms_comp(rms_signal, rms_noise, snr_required)

                                noise_data_comp = noise_data*comp

                                reverbed_noisy_audio = reverbed_audio + noise_data_comp
                                reverbed_noisy_audio = reverbed_noisy_audio / np.max(np.abs(reverbed_noisy_audio))

                                tae = TAE(reverbed_noisy_audio, self.fs, self.sos_lowpass_filter) #Calculo el TAE
                                snr_df.append(snr_required)
                            
                            elif self.add_noise == False:
                                tae = TAE(filtered_speech[i], self.fs, self.sos_lowpass_filter) #Calculo el TAE
                                snr_df.append(nan) 
                            

                            #Guardo los valores:
                            descriptors_df.append([t30, c50, c80, d50])
                            name_df.append(name)
                            band_df.append(band)
                            tae_df.append(list(tae))
                            
                        
                        except (ValueError, NoiseError, Exception) as err:
                            #Paso a otra banda si hubo algún error en el cálculo de los descriptores:
                            print(err.args)
                            continue

                    #self.bar.next()

                else: #Si no es sintética, realizo el cálculo en la original y la aumento:

                    #Realizo el cálculo para la RIR original:

                    #Reverbero el audio:
                    reverbed_audio = fftconvolve(speech_data, rir_data, mode='same') #Reverbero el audio
                    reverbed_audio = reverbed_audio/np.max(np.abs(reverbed_audio))

                    filtered_speech = self.bpfilter.filtered_signals(reverbed_audio) #Filtro la señal de voz por bandas

                    filtered_rir = self.bpfilter.filtered_signals(rir_data) #Filtro la rir por bandas

                    name = f'{speech_name}|{rir_name}|original' #Nombre del archivo en la base de datos

                    for i, band in enumerate(self.bands):

                        try:
                            #Calculo los descriptores:
                            t30, _, _ = tr_lundeby(filtered_rir[i], self.fs, self.max_ruido_dB)
                            c50 = clarity(50, filtered_rir[i], self.fs)
                            c80 = clarity(80, filtered_rir[i], self.fs)
                            d50 = definition(filtered_rir[i], self.fs)

                            #Cálculo del tae:
                            if self.add_noise:
                                #Genero ruido rosa:
                                noise_data = pink_noise(len(filtered_speech[i]))

                                #Agrego ruido para tener SNR entre snr[0] y snr[-1] dB:
                                rms_signal = snr_calculator.rms(filtered_speech[i])
                                rms_noise = snr_calculator.rms(noise_data)

                                snr_required = np.random.uniform(self.snr[0], self.snr[-1], 1)[0]

                                comp = snr_calculator.rms_comp(rms_signal, rms_noise, snr_required)

                                noise_data_comp = noise_data*comp

                                reverbed_noisy_audio = reverbed_audio + noise_data_comp
                                reverbed_noisy_audio = reverbed_noisy_audio / np.max(np.abs(reverbed_noisy_audio))

                                tae = TAE(reverbed_noisy_audio, self.fs, self.sos_lowpass_filter) #Calculo el TAE
                                snr_df.append(snr_required)
                            
                            elif self.add_noise == False:
                                tae = TAE(filtered_speech[i], self.fs, self.sos_lowpass_filter) #Calculo el TAE
                                snr_df.append(nan) 
                            

                            #Guardo los valores:
                            descriptors_df.append([t30, c50, c80, d50])
                            name_df.append(name)
                            band_df.append(band)
                            tae_df.append(list(tae))
                            
                        
                        except (ValueError, NoiseError, Exception) as err:
                            #Paso a otra banda si hubo algún error en el cálculo de los descriptores:
                            #print(err.args)
                            continue

                    #self.bar.next()

                    #Realizo el cálculo para las RIR aumentadas:

                    for TR_var in self.TR_variations: #Por cada aumentación de TR
                        try:
                            #Realizo la aumentación por TR:
                            rir_aug = tr_augmentation(rir_data, self.fs, TR_var, self.bpfilter)
                            rir_aug = rir_aug/np.max(np.abs(rir_aug))

                            #Reverbero el audio en el caso que sea posible aumentar:
                            reverbed_audio = fftconvolve(speech_data, rir_aug, mode='same') #Reverbero el audio
                            reverbed_audio = reverbed_audio/np.max(np.abs(reverbed_audio))  

                            filtered_speech = self.bpfilter.filtered_signals(reverbed_audio) #Filtro la señal de voz por bandas

                            filtered_rir = self.bpfilter.filtered_signals(rir_aug) #Filtro la rir por bandas

                            name = f'{speech_name}|{rir_name}|TR_var_{TR_var}' #Nombre del archivo en la base de datos
                            
                            for i, band in enumerate(self.bands):

                                try:
                                    #Calculo los descriptores:
                                    t30, _, _ = tr_lundeby(filtered_rir[i], self.fs, self.max_ruido_dB)
                                    c50 = clarity(50, filtered_rir[i], self.fs)
                                    c80 = clarity(80, filtered_rir[i], self.fs)
                                    d50 = definition(filtered_rir[i], self.fs)

                                    #Cálculo del tae:

                                    if self.add_noise:
                                        #Genero ruido rosa:
                                        noise_data = pink_noise(len(filtered_speech[i]))

                                        #Agrego ruido para tener SNR entre snr[0] y snr[-1] dB:
                                        rms_signal = snr_calculator.rms(filtered_speech[i])
                                        rms_noise = snr_calculator.rms(noise_data)

                                        snr_required = np.random.uniform(self.snr[0], self.snr[-1], 1)[0]

                                        comp = snr_calculator.rms_comp(rms_signal, rms_noise, snr_required)

                                        noise_data_comp = noise_data*comp

                                        reverbed_noisy_audio = reverbed_audio + noise_data_comp
                                        reverbed_noisy_audio = reverbed_noisy_audio / np.max(np.abs(reverbed_noisy_audio))

                                        tae = TAE(reverbed_noisy_audio, self.fs, self.sos_lowpass_filter) #Calculo el TAE
                                        snr_df.append(snr_required)
                                    
                                    elif self.add_noise == False:
                                        tae = TAE(filtered_speech[i], self.fs, self.sos_lowpass_filter) #Calculo el TAE
                                        snr_df.append(nan) 
                                    

                                    #Guardo los valores:
                                    descriptors_df.append([t30, c50, c80, d50])
                                    name_df.append(name)
                                    band_df.append(band)
                                    tae_df.append(list(tae))

                                    
                                
                                except (ValueError, NoiseError, Exception) as err:
                                    #Paso a otra banda si hubo algún error en el cálculo de los descriptores:
                                    #print(err.args)
                                    continue
                            
                            #self.bar.next()

                        except TrAugmentationError as err:
                            #print(err.args)
                            #self.bar.next()
                            continue

                    for DRR_var in self.DRR_variations: #Por cada aumentación de DRR
                        try:
                            #Realizo la aumentación por TR:
                            rir_aug = drr_aug(rir_data, self.fs, DRR_var)
                            rir_aug = rir_aug/np.max(np.abs(rir_aug))

                            #Reverbero el audio en el caso que sea posible aumentar:
                            reverbed_audio = fftconvolve(speech_data, rir_aug, mode='same') #Reverbero el audio
                            reverbed_audio = reverbed_audio/np.max(np.abs(reverbed_audio))  

                            filtered_speech = self.bpfilter.filtered_signals(reverbed_audio) #Filtro la señal de voz por bandas

                            filtered_rir = self.bpfilter.filtered_signals(rir_aug) #Filtro la rir por bandas

                            name = f'{speech_name}|{rir_name}|DRR_var_{DRR_var}' #Nombre del archivo en la base de datos
                            
                            for i, band in enumerate(self.bands):

                                try:
                                    #Calculo los descriptores:
                                    t30, _, _ = tr_lundeby(filtered_rir[i], self.fs, self.max_ruido_dB)
                                    c50 = clarity(50, filtered_rir[i], self.fs)
                                    c80 = clarity(80, filtered_rir[i], self.fs)
                                    d50 = definition(filtered_rir[i], self.fs)

                                    #Calculo del tae:

                                    if self.add_noise:
                                        #Genero ruido rosa:
                                        noise_data = pink_noise(len(filtered_speech[i]))

                                        #Agrego ruido para tener SNR entre snr[0] y snr[-1] dB:
                                        rms_signal = snr_calculator.rms(filtered_speech[i])
                                        rms_noise = snr_calculator.rms(noise_data)

                                        snr_required = np.random.uniform(self.snr[0], self.snr[-1], 1)[0]

                                        comp = snr_calculator.rms_comp(rms_signal, rms_noise, snr_required)

                                        noise_data_comp = noise_data*comp

                                        reverbed_noisy_audio = reverbed_audio + noise_data_comp
                                        reverbed_noisy_audio = reverbed_noisy_audio / np.max(np.abs(reverbed_noisy_audio))

                                        tae = TAE(reverbed_noisy_audio, self.fs, self.sos_lowpass_filter) #Calculo el TAE
                                        snr_df.append(snr_required)
                                    
                                    elif self.add_noise == False:
                                        tae = TAE(filtered_speech[i], self.fs, self.sos_lowpass_filter) #Calculo el TAE
                                        snr_df.append(nan) 
                                    

                                    #Guardo los valores:
                                    descriptors_df.append([t30, c50, c80, d50])
                                    name_df.append(name)
                                    band_df.append(band)
                                    tae_df.append(list(tae))


                                except (ValueError, NoiseError, Exception) as err:
                                    #Paso a otra banda si hubo algún error en el cálculo de los descriptores:
                                    #print(err.args)
                                    continue

                            #self.bar.next()

                        except Exception as err:
                            #print(err.args)
                            #self.bar.next()
                            continue
        
        #self.bar.next()
        self.i += 1
        print(f'RIR procesada: {rir_name}')
        return [name_df, band_df, tae_df, descriptors_df, snr_df]

    def save_database_multiprocess(self, results):
        
        name_df, band_df, tae_df, descriptors_df, snr_df = [], [], [], [], []

        #Creo la carpeta donde guardo la base de datos:
        os.mkdir(f'cache/{self.db_name}')

        print('Guardando base de datos...')
        
        for result in results:

            #Descompongo los resultados del cálculo con multiprocessing:
            for i in range(len(result[0])):
                name_df.append(result[0][i])
                band_df.append(result[1][i])
                tae_df.append(result[2][i])
                descriptors_df.append(result[3][i])
                snr_df.append(result[4][i])
        
        #Guardo la base de datos en archivos de a 50000 audios:
        buffer = 50000
        tot_cuts = int(len(name_df)/buffer)

        if tot_cuts==0:
            #Genero el dataframe:
            data = {'ReverbedAudio': name_df,
                    'banda': band_df,
                    'tae': tae_df,
                    'descriptors': descriptors_df,
                    'snr': snr_df}
            
            db_df = pd.DataFrame(data)

            db_df.to_pickle(f'cache/{self.db_name}/0.pkl', protocol=5)

            print('Base de datos guardada!')
        
        else:
            for cut in range(tot_cuts):
                data = {'ReverbedAudio': name_df[int(cut*buffer):int((cut+1)*buffer)],
                    'banda': band_df[int(cut*buffer):int((cut+1)*buffer)],
                    'tae': tae_df[int(cut*buffer):int((cut+1)*buffer)],
                    'descriptors': descriptors_df[int(cut*buffer):int((cut+1)*buffer)],
                    'snr': snr_df[int(cut*buffer):int((cut+1)*buffer)]}
            
                db_df = pd.DataFrame(data)

                db_df.to_pickle(f'cache/{self.db_name}/{cut}.pkl', protocol=5)
            
            data = {'ReverbedAudio': name_df[int(tot_cuts*buffer):],
                    'banda': band_df[int(tot_cuts*buffer):],
                    'tae': tae_df[int(tot_cuts*buffer):],
                    'descriptors': descriptors_df[int(tot_cuts*buffer):],
                    'snr': snr_df[int(tot_cuts*buffer):]}
            
            db_df = pd.DataFrame(data)

            db_df.to_pickle(f'cache/{self.db_name}/{tot_cuts}.pkl', protocol=5)

            print('Base de datos guardada!')
    
    def get_database_name(self):
        return self.db_name