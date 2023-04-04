import os
os.chdir('../')

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import random
import gc
import sys
import numpy as np
sys.path.append('code')
#from data_reader import read_dataset

from progress.bar import IncrementalBar
import pickle

def model(filters: list = [32, 18, 8, 4], 
           kernel_size: list = [10, 5, 5, 5], 
           activation: list = ['relu','relu','relu','relu'], 
           pool_size: list = [2,2,2],
           learning_rate: float = 0.001):

    """Se utiliza como modelo una red neruronal convolucional de 4 capas con los siguientes
    valores por defecto:
    1) filters = [32, 18, 8, 4], 
    2) kernel_size = [10, 5, 5, 5], 
    3) activation = ['relu','relu','relu','relu'], 
    4) pool_size = [2,2,2],
    5) learning_rate = 0.001
    """

    import tensorflow as tf
    import tensorflow.keras.layers as tfkl
    
    tf.keras.backend.clear_session()

    audio_in = tfkl.Input((200,1), name = 'Audio de entrada')

    capa_1 = tfkl.Conv1D(filters=filters[0], kernel_size=(kernel_size[0]), activation=activation[0], name='capa1_conv')(audio_in)
    capa_1 = tfkl.MaxPool1D(pool_size=pool_size[0], name='capa1_pool')(capa_1)
    capa_1 = tfkl.BatchNormalization(name = 'Batch_capa1')(capa_1)

    capa_2 = tfkl.Conv1D(filters=filters[1], kernel_size=(kernel_size[1]), activation=activation[1], name='capa2_conv')(capa_1) 
    capa_2 = tfkl.MaxPool1D(pool_size=pool_size[1], name='capa2_pool')(capa_2)
    capa_2 = tfkl.BatchNormalization(name = 'Batch_capa2')(capa_2)
    capa_2 = tfkl.Dropout(0.4, name='capa2_drop')(capa_2)

    capa_3 = tfkl.Conv1D(filters=filters[2], kernel_size=(kernel_size[2]), activation=activation[2], name='capa3_conv')(capa_2) 
    capa_3 = tfkl.MaxPool1D(pool_size=pool_size[2], name='capa3_pool')(capa_3)
    capa_3 = tfkl.BatchNormalization(name = 'Batch_capa3')(capa_3)
    
    capa_4 = tfkl.Conv1D(filters=filters[3], kernel_size=(kernel_size[3]), activation=activation[3], name='capa4_conv')(capa_3)
    capa_4 = tfkl.Flatten()(capa_4)

    tr_pred = tfkl.Dense(4, name='Salida_prediccion')(capa_4)
    
    modelo = tf.keras.Model(inputs=[audio_in], outputs=[tr_pred])
    modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                   loss='mse')
    return modelo

def reshape_data(tae, descriptors):
    """Función para cambiar las dimensiones de los datos para que puedan ser interpretados por Tensorflow.
    """
    tae_list = [[]] * int(len(tae)) #Genero una lista de listas vacías
    descriptors_list = [[]] * int(len(descriptors)) #Genero una lista de listas vacías

    for i in range(len(tae)):
        tae_list[i] = np.array(tae[i]).reshape(-1, 1)
        descriptors_list[i] = np.array(descriptors[i]).reshape(-1, 1)

    X = np.array(tae_list)
    y = np.array(descriptors_list)

    return X, y

def normalize_descriptors(descriptors, y_train, y_test):
    """Luego de hacer el reshape, se usa esta función para normalizar los descriptores
    con su percentil 95"""

    descriptors = list(descriptors)

    #Normalización de los parámetros:
    T30 = [descriptors[i][0][0] for i in range(len(descriptors))]
    C50 = [descriptors[i][1][0] for i in range(len(descriptors))]
    C80 = [descriptors[i][2][0] for i in range(len(descriptors))]
    D50 = [descriptors[i][3][0] for i in range(len(descriptors))]

    T30_perc_95 = np.percentile(T30, 95)
    C50_perc_95 = np.percentile(C50, 95)
    C80_perc_95 = np.percentile(C80, 95)
    D50_perc_95 = np.percentile(D50, 95)

    norm = np.array([T30_perc_95, C50_perc_95, C80_perc_95, D50_perc_95]).reshape(-1, 1)

    y_train = np.array([y_train[i]/norm for i in range(len(y_train))])
    y_test = np.array([y_test[i]/norm for i in range(len(y_test))])

    return y_train, y_test, T30_perc_95, C50_perc_95, C80_perc_95, D50_perc_95

def prediction(blind_estimation_model, X_test, y_test):
    """Función para calcular las predicciones del set de pruebas
    blind_estimation_model: Instancia de entrenamiento del modelo.
    X_test: set de pruebas de los TAE.
    y_test: set de pruebas de los descriptores."""
    
    prediction = []
    bar = IncrementalBar('Predicting values', max = int(len(y_test)))

    for i in range(len(y_test)):
        prediction.append(np.round(blind_estimation_model.predict(X_test[i,:,0].reshape(1,-1,1)),2)[0])
        bar.next()
    bar.finish()

    return prediction

def descriptors_err(prediction, y_test):
    """Función para calcular el error en la predicción de los descriptores."""

    err_t30, err_c50, err_c80, err_d50 = [], [], [], []
    bar = IncrementalBar('Calculating descriptors errors', max = int(len(y_test)))

    for i in range(len(y_test)):
        err_t30.append(np.round(prediction[i][0] - np.round(y_test[i,:,0].reshape(1,-1,1),2).flatten()[0],2))
        err_c50.append(np.round(prediction[i][1] - np.round(y_test[i,:,0].reshape(1,-1,1),2).flatten()[1],2))
        err_c80.append(np.round(prediction[i][2] - np.round(y_test[i,:,0].reshape(1,-1,1),2).flatten()[2],2))
        err_d50.append(np.round(prediction[i][3] - np.round(y_test[i,:,0].reshape(1,-1,1),2).flatten()[3],2))
        bar.next()
    bar.finish()

    return err_t30, err_c50, err_c80, err_d50

def save_exp_data(exp_num, band, blind_estimation_model, history, prediction, 
                  err_t30, err_c50, err_c80, err_d50, 
                  T30_perc_95, C50_perc_95, C80_perc_95, D50_perc_95,
                  X_test, y_test):
    """Función para guardar todos los datos del experimento para poder hacer el análisis en los notebooks"""

    #Creo el directorio donde se va a guardar el experimento si no existe:
    isExist = os.path.exists(f'results/exp{exp_num}')
    if not isExist:
        os.makedirs(f'results/exp{exp_num}')

    #Guardo los pesos del modelo entrenado:
    blind_estimation_model.save_weights(f'results/exp{exp_num}/weights_{band}.h5')

    #Guardo en un diccionario los resultados del modelo:
    results_dic = {'loss': history.history['loss'],
                   'val_loss': history.history['val_loss'],
                   'prediction': prediction,
                   'err_t30': err_t30,
                   'err_c50': err_c50,
                   'err_c80': err_c80,
                   'err_d50': err_d50,
                   'T30_perc_95': T30_perc_95,
                   'C50_perc_95': C50_perc_95,
                   'C80_perc_95': C80_perc_95,
                   'D50_perc_95': D50_perc_95,
                   'X_test': X_test,
                   'y_test': y_test}

    with open(f'results/exp{exp_num}/results_{band}.pickle', 'wb') as handle:
        pickle.dump(results_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f'Resultados guardados en la carpeta: results/exp{exp_num}')

def read_dataset(band, db_name, sample_frac=1.0, random_state=None, type_data='train'):
    """Script para leer la base de datos:
    band (int): banda a leer.
    max_ruido_dB (int): máximo ruido permitido en la base de datos a leer.
    noise (bool): 'True' si se quiere leer la base de datos con ruido, sino 'False'.
    sample_frac (float): valor entre 0 y 1 que indica el porcentaje a leer de la base de datos.
    random_state (int): seed para poder reproducir un random de lectura de la base de datos."""

    partitions = os.listdir(f'cache/{db_name}')

    bar = IncrementalBar('Reading data base', max = len(partitions))

    db = pd.DataFrame()
    for partition in partitions:
        #Leo la base de datos:
        aux_df = pd.read_pickle(f'cache/{db_name}/{partition}')
        #Filtro por la banda y por si es train o test:
        db = db.append(aux_df.loc[(aux_df.banda == band) & (aux_df.type_data == type_data)], ignore_index=True)
        bar.next()
    
    db = db.sample(frac=sample_frac, random_state=random_state)
    bar.finish()

    return db

def create_dataset(band, db_name, sample_frac=1.0, random_state=None):
    """Script para crear la base de datos a partir de una existente:
    band (int): banda a leer.
    max_ruido_dB (int): máximo ruido permitido en la base de datos a leer.
    noise (bool): 'True' si se quiere leer la base de datos con ruido, sino 'False'.
    sample_frac (float): valor entre 0 y 1 que indica el porcentaje a leer de la base de datos.
    random_state (int): seed para poder reproducir un random de lectura de la base de datos."""

    partitions = os.listdir(f'cache/{db_name}')

    bar = IncrementalBar('Reading data base', max = len(partitions))

    db = pd.DataFrame()
    for partition in partitions:
        #Leo la base de datos:
        aux_df = pd.read_pickle(f'cache/{db_name}/{partition}')
        #Filtro por la banda y solo me quedo con las RIRs originales:
        db = db.append(aux_df.loc[aux_df.banda == band], ignore_index=True)
        bar.next()
    
    db = db.sample(frac=sample_frac, random_state=random_state)
    bar.finish()

    return db

if __name__ == '__main__':
    
    # Saco data del experimento 1, en este caso solo quiero entrenar con RIRs reales:

    import time
    start_time = time.time()

    exp_num = 6 #Número del experimento

    seed = 2222
    random.seed(seed)

    bands = [125, 250, 500, 1000, 2000, 4000, 8000] #Bandas a analizar
    filter_type = 'octave band' #Tipo de filtro a utilizar: 'octave band' o 'third octave band'
    fs = 16000 #Frecuencia de sampleo de los audios.
    order = 4 #Orden del filtro
    max_ruido_dB = -45 #Criterio de aceptación de ruido para determinar si una RIR es válida o no
    add_noise = False #Booleano para definir si agregar ruido rosa o no a la base de datos.
    snr = [-5, 20] #Valores de SNR que tendrían los audios si se les agrega ruido
    tr_aug = [0.2, 3.1, 0.1] #Aumentar los valores de TR de 0.2 a 3 s con pasos de 0.1 s
    drr_aug = [-6, 19, 1] #Aumentar los valores de DRR de -6 a 18 dB con pasos de 1 dB

    #Parámetros para la lectura de la base de datos:
    sample_frac = 1.0 #Fracción de la data a leer
    random_state = seed #Inicializador del generador de números random

    # Modelo:

    #Parámetros de la red:
    filters = [32, 18, 8, 4] 
    kernel_size = [10, 5, 5, 5] 
    activation = ['relu','relu','relu','relu'] 
    pool_size = [2,2,2]
    learning_rate = 0.001

    #Parámetros de entrenamiento:
    validation_split = 0.1 
    batch_size = 256
    epochs = 500

    db_name = f'base_de_datos_{max_ruido_dB}_noise_{add_noise}_traug_{tr_aug[0]}_{tr_aug[1]}_{tr_aug[2]}_drraug_{drr_aug[0]}_{drr_aug[1]}_{drr_aug[2]}_snr_{snr[0]}_{snr[-1]}'

    new_db_name = 'exp_todos_' + db_name

    # Genero la nueva base de datos en caso de que no exista:

    available_cache_files = os.listdir('cache')

    db_exists = False
    for available in available_cache_files:
        if new_db_name in available:
            db_exists = True
    
    if db_exists:
        print('La base de datos ya existe!')

    else:

        db = pd.DataFrame()
        train = 0.8
        test = 0.2
        for band in bands:
            print(band)
            
            db_aux = create_dataset(band, db_name, 1.0, seed)

            db_check = db_aux[db_aux.ReverbedAudio.str.lower().str.contains("original")]
            db_check = db_check[~db_check.ReverbedAudio.str.lower().str.contains("sintetica")]

            print(len(db_check))
            
            if len(db_check) == 0:
                del db_check
                gc.collect()
                continue
            
            else:

                del db_check
                gc.collect()

                audios_test = read_dataset(band, f'exp_reales_base_de_datos_{max_ruido_dB}_noise_{add_noise}_traug_{tr_aug[0]}_{tr_aug[1]}_{tr_aug[2]}_drraug_{drr_aug[0]}_{drr_aug[1]}_{drr_aug[2]}_snr_{snr[0]}_{snr[-1]}', 
                                           1.0, seed, 'test').ReverbedAudio.tolist()

                audios_reales = read_dataset(band, f'exp_reales_base_de_datos_{max_ruido_dB}_noise_{add_noise}_traug_{tr_aug[0]}_{tr_aug[1]}_{tr_aug[2]}_drraug_{drr_aug[0]}_{drr_aug[1]}_{drr_aug[2]}_snr_{snr[0]}_{snr[-1]}', 
                                           1.0, seed, 'train').ReverbedAudio.tolist()

                audios = db_aux.ReverbedAudio.tolist()

                # Me quedo solo con los reales sin aumentar:
                audios_reales = random.sample(audios_reales, k=int(len(audios_reales)/3))

                tot_reales = len(audios_reales)

                # Me quedo con las sinteticas sin aumentar:
                audios_sinteticas = [audio for audio in audios if 'sintetica' in audio]
                audios_sinteticas = random.sample(audios_sinteticas, k=int(tot_reales))

                # Me quedo con las aumentadas por DRR:
                audios_drr = [audio for audio in audios if 'DRR_var_' in audio]
                audios_drr = random.sample(audios_drr, k=int(tot_reales/2))

                # Me quedo con las aumentadas por TR:
                audios_tr = [audio for audio in audios if 'TR_var_' in audio]
                audios_tr = random.sample(audios_tr, k=int(tot_reales/2))

                # Junto todos los audios:

                audios = audios_reales + audios_sinteticas + audios_drr + audios_tr

                #Les doy un orden random a los audios de train:
                audios_train = random.sample(audios, k=int(len(audios)))

                #Lo agrego a train:
                idx_train = []
                for audio in audios_train:
                    idx = db_aux[db_aux['ReverbedAudio'] == audio].index.values
                    idx_train.append(int(idx[0]))
                    db_aux['type_data'][idx] = 'train'
                    
                
                #Lo agrego a test:
                idx_test = []
                for audio in audios_test:
                    idx = db_aux[db_aux['ReverbedAudio'] == audio].index.values
                    idx_test.append(int(idx[0]))
                    db_aux['type_data'][idx] = 'test'
                
                idx = idx_train + idx_test
                idx.sort()

                db = db.append(db_aux.iloc[idx], ignore_index=True)
                
                

        #Guardo la base de datos:

        os.mkdir(f'cache/{new_db_name}')

        #Guardo la base de datos en archivos de a 50000 audios:
        buffer = 50000
        tot_cuts = int(len(db)/buffer)

        if tot_cuts==0:

            db.to_pickle(f'cache/{new_db_name}/0.pkl', protocol=5)

            print('Base de datos guardada!')
        
        else:
            for cut in range(tot_cuts):
            
                db_chunck = db[int(cut*buffer):int((cut+1)*buffer)]

                db_chunck.to_pickle(f'cache/{new_db_name}/{cut}.pkl', protocol=5)
            
            
            db_chunck = db[int(tot_cuts*buffer):]

            db_chunck.to_pickle(f'cache/{new_db_name}/{tot_cuts}.pkl', protocol=5)

            print('Base de datos guardada!')
        

    #Entrenamiento:
    for band in bands:
        print(f'\nInicio entrenamiento de la banda {band} Hz:')

        #Leo la fracción de datos especificados para la banda seleccionada:
        db_train = read_dataset(band, new_db_name, sample_frac, random_state, type_data='train')
        db_test = read_dataset(band, new_db_name, sample_frac, random_state, type_data='test')

        if (len(db_train) == 0) or (len(db_test) == 0):
            continue

        else:
            tae_train = list(db_train.tae.to_numpy())
            tae_test = list(db_test.tae.to_numpy())
            descriptors_train = list(db_train.descriptors.to_numpy())
            descriptors_test = list(db_test.descriptors.to_numpy())

            #Separo en train y test y les doy formato:
            X_train, y_train = reshape_data(tae_train, descriptors_train)
            X_test, y_test = reshape_data(tae_test, descriptors_test)

            #Normalizo según el percentil 95 de cada descriptor:
            descriptors = np.concatenate((y_train, y_test), axis=0)
            y_train, y_test, T30_perc_95, C50_perc_95, C80_perc_95, D50_perc_95 = normalize_descriptors(descriptors, y_train, y_test)

            #Instancio el modelo:
            blind_estimation_model = model(filters, kernel_size, activation, pool_size, learning_rate)
            #blind_estimation_model.summary()

            #Entreno el modelo:
            history = blind_estimation_model.fit(x = X_train, y = y_train, 
                                                validation_split = validation_split, 
                                                batch_size = batch_size, 
                                                epochs = epochs)

            #Realizo las predicciones del modelo:
            predict = prediction(blind_estimation_model, X_test, y_test)

            #Calculo del error de los descriptores:
            err_t30, err_c50, err_c80, err_d50 = descriptors_err(predict, y_test)

            #Guardo los datos del experimento:
            save_exp_data(exp_num, band, blind_estimation_model, history, predict, 
                        err_t30, err_c50, err_c80, err_d50, 
                        T30_perc_95, C50_perc_95, C80_perc_95, D50_perc_95,
                        X_test, y_test)

            # Elimino estas variables de memoria:
            del db_train, db_test, tae_train, tae_test, descriptors, X_train, X_test, y_train, y_test
            del descriptors_train, descriptors_test
            del T30_perc_95, C50_perc_95, C80_perc_95, D50_perc_95
            del blind_estimation_model, history, predict, err_t30, err_c50, err_c80, err_d50
            gc.collect()

    """
    Pasos a seguir:

        1) Leer la base de datos según lo que necesito
        2) Generar una nueva con solo los reales y volver a clasificar lo de train y test
        3) Verificar según los nombres únicos que aparezcan solo en los tae de train y solo los de test
    
    """