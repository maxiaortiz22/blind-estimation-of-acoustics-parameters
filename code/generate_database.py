from parameters_calculation import clarity
from parameters_calculation import definition
from parameters_calculation import tr_lundeby, NoiseError
from parameters_calculation import BandpassFilter
from parameters_calculation import TAE
from librosa import load
import pandas as pd
from os import listdir
from math import nan
from scipy.signal import butter, fftconvolve
from parameters_calculation import pink_noise
import numpy as np
import parameters_calculation as snr_calculator
from parameters_calculation import tr_augmentation, TrAugmentationError
from parameters_calculation import drr_aug, get_DRR
import os
import random
from datetime import datetime

class DataBase():
    """Clase para generar el cálculo de la base de datos para entrenar la red."""

    __slots__ = ('speech_files', 'rir_files', 'tot_sinteticas', 'to_augmentate', 'bands', 'filter_type', 
                 'fs', 'max_ruido_dB', 'order', 'add_noise', 'snr', 'TR_aug', 'DRR_aug', 'db_name', 'cutoff',
                 'sos_lowpass_filter', 'TR_variations', 'DRR_variations', 'bpfilter')
    
    def __init__(self, speech_files, rir_files, tot_sinteticas, to_augmentate,  bands, filter_type, fs, max_ruido_dB, order, add_noise, snr, TR_aug, DRR_aug):
        self.speech_files = speech_files
        self.rir_files = rir_files
        self.tot_sinteticas = tot_sinteticas
        self.to_augmentate = to_augmentate
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
            tae_df, snr_df, drr_df = [], [], []

            random.seed(datetime.now()) #Cambio la seed por el horario!
            DRR_tr_aug = random.sample(list(self.TR_variations), k=5)
            
            for speech_file in self.speech_files: #Por cada audio de voz

                speech_name = speech_file.split('.wav')[0]
                speech_data, _ = load(f'data/Speech/{speech_file}', sr=self.fs, duration=5.0)
                speech_data = speech_data/np.max(np.abs(speech_data))
                

                #Obtengo la rir
                rir_name = rir_file.split('.wav')[0]
                rir_data, _ = load(f'data/RIRs/{rir_file}', sr=self.fs)
                rir_data = rir_data/np.max(np.abs(rir_data))

                if ('sintetica' in rir_name) or not any(rir_name in s for s in self.to_augmentate): #Si se trata de una RIR sintética o de una de las RIRs que no se eligirieron para aumentar

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
                            drr, _, _ = get_DRR(filtered_rir[i], self.fs)

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
                            drr_df.append(drr)
                            
                        
                        except (ValueError, NoiseError, Exception) as err:
                            #Paso a otra banda si hubo algún error en el cálculo de los descriptores:
                            print(err.args)
                            continue

                else: #Realizo el cálculo en la original y la aumento para las seleccionadas:

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
                            drr, _, _ = get_DRR(filtered_rir[i], self.fs)

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
                            drr_df.append(drr)
                            
                        
                        except (ValueError, NoiseError, Exception) as err:
                            #Paso a otra banda si hubo algún error en el cálculo de los descriptores:
                            #print(err.args)
                            continue

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
                                    drr, _, _ = get_DRR(filtered_rir[i], self.fs)

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
                                    drr_df.append(drr)

                                    
                                
                                except (ValueError, NoiseError, Exception) as err:
                                    #Paso a otra banda si hubo algún error en el cálculo de los descriptores:
                                    #print(err.args)
                                    continue

                            #Al TR aumentado elegido aleatoriamente le aumento el DRR también:

                            if any([True for val in DRR_tr_aug if val == TR_var]): 

                                rir_tr_aug = rir_aug
                                print(f'TR de los valores a aumentarle el DRR: {DRR_tr_aug}')

                                for DRR_var in self.DRR_variations: #Por cada aumentación de DRR
                                    print('Aumentando DRR')
                                    try:
                                        #Realizo la aumentación por TR:
                                        rir_aug = drr_aug(rir_tr_aug, self.fs, DRR_var)
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
                                                drr, _, _ = get_DRR(filtered_rir[i], self.fs)

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
                                                drr_df.append(drr)


                                            except (ValueError, NoiseError, Exception) as err:
                                                #Paso a otra banda si hubo algún error en el cálculo de los descriptores:
                                                #print(err.args)
                                                continue

                                    except Exception as err:
                                        #print(err.args)
                                        continue
                        
                        except TrAugmentationError as err:
                            #print(err.args)
                            continue

        print(f'RIR procesada: {rir_name}')
        return [name_df, band_df, tae_df, descriptors_df, snr_df, drr_df]

    def save_database_multiprocess(self, results):
        
        name_df, band_df, tae_df, descriptors_df, snr_df, drr_df = [], [], [], [], [], []

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
                drr_df.append(result[5][i])
        
        #Guardo la base de datos en archivos de a 50000 audios:
        buffer = 50000
        tot_cuts = int(len(name_df)/buffer)

        if tot_cuts==0:
            #Genero el dataframe:
            data = {'ReverbedAudio': name_df,
                    'banda': band_df,
                    'tae': tae_df,
                    'descriptors': descriptors_df,
                    'snr': snr_df,
                    'drr': drr_df}
            
            db_df = pd.DataFrame(data)

            db_df.to_pickle(f'cache/{self.db_name}/0.pkl', protocol=5)

            print('Base de datos guardada!')
        
        else:
            for cut in range(tot_cuts):
                data = {'ReverbedAudio': name_df[int(cut*buffer):int((cut+1)*buffer)],
                    'banda': band_df[int(cut*buffer):int((cut+1)*buffer)],
                    'tae': tae_df[int(cut*buffer):int((cut+1)*buffer)],
                    'descriptors': descriptors_df[int(cut*buffer):int((cut+1)*buffer)],
                    'snr': snr_df[int(cut*buffer):int((cut+1)*buffer)],
                    'drr': drr_df[int(cut*buffer):int((cut+1)*buffer)]}
            
                db_df = pd.DataFrame(data)

                db_df.to_pickle(f'cache/{self.db_name}/{cut}.pkl', protocol=5)
            
            data = {'ReverbedAudio': name_df[int(tot_cuts*buffer):],
                    'banda': band_df[int(tot_cuts*buffer):],
                    'tae': tae_df[int(tot_cuts*buffer):],
                    'descriptors': descriptors_df[int(tot_cuts*buffer):],
                    'snr': snr_df[int(tot_cuts*buffer):],
                    'drr': drr_df[int(tot_cuts*buffer):]}
            
            db_df = pd.DataFrame(data)

            db_df.to_pickle(f'cache/{self.db_name}/{tot_cuts}.pkl', protocol=5)

            print('Base de datos guardada!')
    
    def get_database_name(self):
        return self.db_name