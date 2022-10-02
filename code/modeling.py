import tensorflow as tf
import tensorflow.keras.layers as tfkl
import numpy as np
from progress.bar import IncrementalBar
import os
import pickle

def model(filters=[32, 18, 8, 4], 
           kernel_size=[10, 5, 5, 5], 
           activation=['relu','relu','relu','relu'], 
           pool_size=[2,2,2],
           learning_rate=0.001):

    """Se utiliza como modelo una red neruronal convolucional de 4 capas con los siguientes
    valores por defecto:
    1) filters = [32, 18, 8, 4], 
    2) kernel_size = [10, 5, 5, 5], 
    3) activation = ['relu','relu','relu','relu'], 
    4) pool_size = [2,2,2],
    5) learning_rate = 0.001
    """
    
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
        tae_list[i] = tae[i].reshape(-1, 1)
        descriptors_list[i] = descriptors[i].reshape(-1, 1)

    X = np.array(tae_list)
    y = np.array(descriptors_list)

    return X, y

def normalize_descriptors(descriptors):
    """Luego de hacer el reshape, se usa esta función para normalizar los descriptores
    con su percentil 95"""
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

    descriptors = np.array([descriptors[i]/norm for i in range(len(descriptors))])

    return descriptors, T30_perc_95, C50_perc_95, C80_perc_95, D50_perc_95

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

def save_exp_data(exp_num, blind_estimation_model, history, prediction, 
                  err_t30, err_c50, err_c80, err_d50, 
                  T30_perc_95, C50_perc_95, C80_perc_95, D50_perc_95):
    """Función para guardar todos los datos del experimento para poder hacer el análisis en los notebooks"""

    #Creo el directorio donde se va a guardar el experimento si no existe:
    isExist = os.path.exists(f'results/exp{exp_num}')
    if not isExist:
        os.makedirs(f'results/exp{exp_num}')

    #Guardo los pesos del modelo entrenado:
    blind_estimation_model.save_weights(f'results/exp{exp_num}/weights.h5')

    #Guardo en un diccionario los resultados del modelo:
    results_dic = {'history': history,
                   'prediction': prediction,
                   'err_t30': err_t30,
                   'err_c50': err_c50,
                   'err_c80': err_c80,
                   'err_d50': err_d50,
                   'T30_perc_95': T30_perc_95,
                   'C50_perc_95': C50_perc_95,
                   'C80_perc_95': C80_perc_95,
                   'D50_perc_95': D50_perc_95}

    with open(f'results/exp{exp_num}/results.pickle', 'wb') as handle:
        pickle.dump(results_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f'Resultados guardados en la carpeta: results/exp{exp_num}')






####### de otro proyecto ########:
"""
from sklearn.metrics import classification_report
from .utils import flatten
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def modeling(data,features,model_config,folds):
    results = []
    
    for fold in folds:
        val_index=data[data['fold']==fold].index                
        train_index=data[data['fold']!=fold].index
        
        x_train=features.loc[train_index].to_numpy()
        y_train=data.loc[train_index]['label']

        x_val=features.loc[val_index].to_numpy()
        y_val=data['label'].loc[val_index]

        model = train(x_train, y_train, model_config)
        
        scores_val, predictions_val= eval(model, x_val)        
        metrics_val = compute_metrics(y_val,predictions_val)
        results.append({'scores':scores_val,'predictions':predictions_val,'metrics':metrics_val,'set':'val'})
        scores_train, predictions_train = eval(model, x_train)        
        metrics_train = compute_metrics(y_train,predictions_train)
        results.append({'scores':scores_train,'predictions':predictions_train,'metrics':metrics_train,'set':'train'})
    
    return results

def train(x_train, y_train, model_config):

    if model_config['model']=='random_forest':
        
        model=RandomForestClassifier(**model_config['parameters'])
        model.fit(x_train,y_train)

    return model

def eval(model,x):
    
    scores=model.predict_proba(x)
    classes=model.classes_
    predictions=classes[scores.argmax(1)]
    
    return scores, predictions

def compute_metrics(y_true, y_pred):
    metrics = flatten(classification_report(y_true, y_pred, output_dict=True))
    return metrics

def part_data(dummy,a,b,c):
    return dummy
"""