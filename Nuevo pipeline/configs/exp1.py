"""
Experiment description
"""
from itertools import product

# Global

seed = 123456
results_path = "results/exp1"

# Data

# previo a este paso, crear link un simbólico dentro de este directorio al directorio real
data_path = 'data/speechcommands'
frac_data = 0.1

# Features

# el ejemplo de tupla sería que se concatenan las mfcc con las deltas y doble deltas
features = {
    'mfcc': {'feat': 'mfcc', 'parameters': {'n_mfcc': 13}},
    # 'melspectrogram': {'feat': 'melspectrogram', 'parameters': {'frame_length_ms': 25, 'frame_hop_ms': 10}},
    'mfccdeltas': {'feat': ('mfcc', 'dmfcc', 'ddmfcc'), 'parameters': {'n_mfcc': 13}}
}

# Partitioning

# folds estratificados en label y speaker independent (idealmente)

n_folds = 5
stratify_column = 'label'
independent_column = 'speaker'

# Modeling

# a mano
# models = {
#     'random_forest_1': {
#         'model': 'random_forest',
#         'parameters': {'n_estimators': 50}},
#     'random_forest_2': {
#         'model': 'random_forest',
#         'parameters': {'n_estimators': 100}}
# }

models = {}
# iterando
for i, n_estimators in enumerate([50, 100]):
    models[f'random_forest_{i}'] = {
        'model': 'random_forest',
        'parameters': {'n_estimators': n_estimators}}


# Run

# opción 1, la iteración se resuelve más adelante
# run = {
#     'xval': range(n_folds),
#     'models': models,
#     'features': features
# }

# opción 2, se resuelve la iteración de features y modelos ahora, xval queda para más adelante

xval = range(n_folds)

runs = {}
for i, (feat, model) in enumerate(product(features, models)):
    runs[i] = {
        'xval': xval,
        'models': models[model],
        'features': features[feat]
    }
