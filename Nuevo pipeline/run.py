
import argparse
from code.utils import import_configs_objs, save_results
from code.data_reader import read_dataset
from code.modeling import modeling
from code.partition import part_data
from code.features import extract_features
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
    print(config)

    # Lectura de datos
    #data = read_dataset(config['data_path'], config['frac_data'], config['seed'])
    #data = part_data(data,config["n_folds"],config["stratify_column"],config["independent_column"])
    
    # Diccionario con información para las corridas
    #runs = config.pop('runs')

    # Inicio del modelado
    #model_results = []
    #for i, run in runs.items():
    #    print('Extract features')
    #    features = extract_features(data,**run["features"])
    #    print('Modeling')
    #    this_run_results = modeling(data,features,run["models"],run["xval"])
        
    #    for result in this_run_results:
    #        result["run"] = run

    #    model_results.extend(this_run_results)
    
    # Guardado de los resultados
    #save_results(model_results,config["results_path"])




if __name__ == "__main__":

    # 1) Lectura de los argumentos
    kwargs = parse_args()

    # 2) Función principal
    main(**kwargs)
