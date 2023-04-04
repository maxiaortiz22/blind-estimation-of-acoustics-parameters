import sys; sys.path.append('code')
import os
import argparse
from code import import_configs_objs
from code import DataBase
from code import read_dataset
from sklearn.model_selection import train_test_split
from code import model, reshape_data, normalize_descriptors, prediction, descriptors_err, save_exp_data
import concurrent.futures
import gc
from numpy import concatenate
from warnings import filterwarnings
filterwarnings("ignore")

def parse_args():
    """Función para parsear los argumentos de línea de comando"""

    # Inicializo el argparse
    parser = argparse.ArgumentParser()

    # Lista de argumentos por línea de comando
    parser.add_argument(
        "--config", help="Config file with the experiment configurations")

    # Convierto a diccionario
    command_line_args = vars(parser.parse_args())
    return command_line_args


def main(**kwargs):
    """Función principal"""

    # Carga de los objetos del config en un diccionario
    config_path = kwargs.pop("config")
    print(config_path)
    config = import_configs_objs(config_path)
    #print(config)

    #Creo la base de datos si no existe esta configuración en la carpeta cache:

    database = DataBase(config['files_speech_train'], config['files_speech_test'], config['files_rirs'], config['tot_sinteticas'], config['to_augmentate'], 
                        config['rirs_for_training'], config['rirs_for_testing'], config['bands'], config['filter_type'], config['fs'], config['max_ruido_dB'], 
                        config['order'], config['add_noise'], config['snr'], config['tr_aug'], config['drr_aug'])

    db_name = database.get_database_name()

    db_exists = False
    for folder in os.listdir('cache/'):
        if db_name in folder:
            db_exists = True

    if db_exists:
        print('Base de datos calculada')
    
    else:
        with concurrent.futures.ProcessPoolExecutor() as executor:

            results = executor.map(database.calc_database_multiprocess, config['files_rirs'])

        database.save_database_multiprocess(results)
    
    del database #Elimino el objeto de la base de datos de memoria

    gc.collect() #Llamo al garbage collector de python
        
    #Entrenamiento:
    for band in config['bands']:
        print(f'\nInicio entrenamiento de la banda {band} Hz:')

        #Leo la fracción de datos especificados para la banda seleccionada:
        db_train = read_dataset(band, db_name, config['sample_frac'], config['random_state'], type_data='train')
        db_test = read_dataset(band, db_name, config['sample_frac'], config['random_state'], type_data='test')

        tae_train = list(db_train.tae.to_numpy())
        tae_test = list(db_test.tae.to_numpy())
        descriptors_train = list(db_train.descriptors.to_numpy())
        descriptors_test = list(db_test.descriptors.to_numpy())

        #Separo en train y test y les doy formato:
        X_train, y_train = reshape_data(tae_train, descriptors_train)
        X_test, y_test = reshape_data(tae_test, descriptors_test)

        #Normalizo según el percentil 95 de cada descriptor:
        descriptors = concatenate((y_train, y_test), axis=0)
        y_train, y_test, T30_perc_95, C50_perc_95, C80_perc_95, D50_perc_95 = normalize_descriptors(descriptors, y_train, y_test)

        #Instancio el modelo:
        blind_estimation_model = model(config['filters'], config['kernel_size'], config['activation'], 
                                       config['pool_size'], config['learning_rate'])
        #blind_estimation_model.summary()

        #Entreno el modelo:
        history = blind_estimation_model.fit(x = X_train, y = y_train, 
                                             validation_split = config['validation_split'], 
                                             batch_size = config['batch_size'], 
                                             epochs = config['epochs'])

        #Realizo las predicciones del modelo:
        predict = prediction(blind_estimation_model, X_test, y_test)

        #Calculo del error de los descriptores:
        err_t30, err_c50, err_c80, err_d50 = descriptors_err(predict, y_test)

        #Guardo los datos del experimento:
        save_exp_data(config['exp_num'], band, blind_estimation_model, history, predict, 
                      err_t30, err_c50, err_c80, err_d50, 
                      T30_perc_95, C50_perc_95, C80_perc_95, D50_perc_95,
                      X_test, y_test)

        # Elimino estas variables de memoria:
        del db_train, db_test, tae_train, tae_test, descriptors, X_train, X_test, y_train, y_test
        del descriptors_train, descriptors_test
        del T30_perc_95, C50_perc_95, C80_perc_95, D50_perc_95
        del blind_estimation_model, history, predict, err_t30, err_c50, err_c80, err_d50
        gc.collect()


if __name__ == "__main__":

    # 1) Lectura de los argumentos
    kwargs = parse_args()

    # 2) Función principal
    main(**kwargs)
