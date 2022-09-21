"""
audio(arr) -->[func]--> features(arr)

* noise reduction
* silence removal
* padding

"""
from bdb import set_trace
from webbrowser import get
import librosa 
import pandas as pd


# def extract_features(data,**features_config):
#     feat_name = features_config.pop("feat")
#     params = features_config.pop("parameters")
#     # TODO: implementar la extracción de features
#     features = None
#     return features

def get_mfcc(audio, sr, **params):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, **params)
    return mfcc


def get_stft(audio, sr, n_fft=1024, win_size=1024, hop_size=512, window='hann'):
    stft = librosa.stft(audio, n_fft=n_fft, win_length=win_size,
                        hop_length=hop_size, window=window)
    return stft

def minmax_norm():
    pass

def log_scale():
    pass

def summary(features, aggfuncs=('mean', 'std', 'max', 'min')):
    return features.groupby('name').agg(aggfuncs)

def extract_features(data, **params):
    list_features = []
    deltas = isinstance(params['feat'], tuple)
    if deltas:
        feature_type = params['feat'][0]
    else:
        feature_type = params['feat']
    for _, row in data.iterrows():
        audio, sr = librosa.load(row.audio_filename, sr=None)
        if feature_type == 'mfcc':
            feat = get_mfcc(audio, sr, **params["parameters"])
            feat_df = pd.DataFrame(feat.T, columns=[f'mfcc_{i}' for i in range(params['parameters']['n_mfcc'])])
        if feature_type == 'stft':
            feat = get_stft(audio, sr, **params)
            feat_df = pd.DataFrame(feat.T, columns=[f'freq_bin_{i}' for i in range(feat.shape[0])])
        if deltas:
            base_df = feat_df
            for d in range(len(params['feat'])-1):
                delta_df = pd.DataFrame(librosa.feature.delta(base_df, order = d+1))
                feat_df = pd.concat([feat_df, delta_df], axis=1)
        #import pdb; pdb.set_trace()
        feat_df['name'] = row.name
        list_features.append(feat_df)

    return summary(pd.concat(list_features))


