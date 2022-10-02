import sys
sys.path.append('code')
import argparse
from code.utils import import_configs_objs
from code.generate_database import calc_rir_descriptors, calc_tae
from code.data_reader import read_dataset
from sklearn.model_selection import train_test_split
from code.modeling import model, reshape_data, normalize_descriptors, prediction, descriptors_err, save_exp_data
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
    calc_rir_descriptors(config['files_rirs'], config['bands'], config['filter_type'], 
                         config['fs'], config['order'], config['max_ruido_dB'])

    calc_tae(config['files_voices'], config['bands'], config['filter_type'], config['fs'], 
             config['max_ruido_dB'], config['add_noise'], config['snr'])

    #Entrenamiento:
    for band in config['bands']:
        print(f'\nInicio entrenamiento de la banda {band} Hz:')

        #Leo la fracción de datos especificados para la banda seleccionada:
        db = read_dataset(band, config['max_ruido_dB'], config['add_noise'], config['sample_frac'], config['random_state'])

        tae = list(db.tae.to_numpy())
        descriptors = list(db.descriptors.to_numpy())

        tae, descriptors = reshape_data(tae, descriptors)

        #Separo en train y test:
        X_train, X_test, y_train, y_test = train_test_split(tae, descriptors, test_size=config['test'], random_state=config['random_state'])

        #Normalizo según el percentil 95 de cada descriptor:
        y_train, y_test, T30_perc_95, C50_perc_95, C80_perc_95, D50_perc_95 = normalize_descriptors(descriptors, y_train, y_test)

        #Instancio el modelo:
        blind_estimation_model = model()
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
                      T30_perc_95, C50_perc_95, C80_perc_95, D50_perc_95)


if __name__ == "__main__":

    # 1) Lectura de los argumentos
    kwargs = parse_args()

    # 2) Función principal
    main(**kwargs)
