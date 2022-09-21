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
