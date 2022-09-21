import collections
from importlib.machinery import SourceFileLoader
import os
from types import ModuleType
import pandas as pd

def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def import_configs_objs(config_file):
    """Importación dinámica del archivo de configuraciones"""

    if config_file is None:
        raise ValueError("No config path")

    # Ejecuto el archivo del path config_file
    loader = SourceFileLoader('config', config_file)
    mod = ModuleType(loader.name)
    loader.exec_module(mod)

    # Convierto a un diccionario y elimino todo lo que no es configuraciones
    config_objs = vars(mod)
    config_objs.pop("__name__")
    config_objs.pop("__doc__")
    config_objs.pop("__package__")
    config_objs.pop("__loader__")
    config_objs.pop("__spec__")
    config_objs.pop("__builtins__")

    return config_objs


def save_results(model_results,results_path):
    pd.DataFrame(model_results).to_pickle(os.path.join(results_path,"output.pkl"))
