import argparse
from code.utils import import_configs_objs
from code.generate_database import calc_rir_descriptors, calc_tae
from code.data_reader import read_dataset
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

    #Leo la base de datos:
    db = read_dataset(config['band'], config['max_ruido_dB'], config['noise'], config['sample_frac'], config['random_state'])



if __name__ == "__main__":

    # 1) Lectura de los argumentos
    kwargs = parse_args()

    # 2) Función principal
    main(**kwargs)
