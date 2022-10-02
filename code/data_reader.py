import pandas as pd

def read_dataset(band, max_ruido_dB, noise, sample_frac=1.0, random_state=None):
    """Script para leer la base de datos:
    band (int): banda a leer.
    max_ruido_dB (int): máximo ruido permitido en la base de datos a leer.
    noise (bool): 'True' si se quiere leer la base de datos con ruido, sino 'False'.
    sample_frac (float): valor entre 0 y 1 que indica el porcentaje a leer de la base de datos.
    random_state (int): seed para poder reproducir un random de lectura de la base de datos."""

    if noise:
        #Leo la base de datos con ruido:
        db = pd.read_pickle(f'cache/base_de_datos_ruido_{max_ruido_dB}.pkl')
        #Filtro por la banda y la fracción de datos que quiero:
        db = db.loc[db.banda == band].sample(frac=sample_frac, random_state=random_state)

        return db
    
    else:

        #Leo la base de datos sin ruido:
        db = pd.read_pickle(f'cache/base_de_datos_{max_ruido_dB}.pkl')
        #Filtro por la banda y la fracción de datos que quiero:
        db = db.loc[db.banda == band].sample(frac=sample_frac, random_state=random_state)

        return db
