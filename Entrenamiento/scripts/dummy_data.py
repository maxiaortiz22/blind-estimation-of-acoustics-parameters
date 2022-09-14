"""
Genero dummy data para probar modelos.
La frecuencia de sampleo esta hardcodeada!
"""
import glob
import librosa

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.signal import butter, sosfilt, lfilter, fftconvolve, hilbert


def generate_rir(tr, fs):
    DURACION = 3 #segundos
    
    t = np.linspace(0, DURACION, int(DURACION*fs))
    y = np.e**((-6.9*t)/tr)
    n = np.random.normal(0, 1, y.shape)
    y = y*n
    return y / np.max(np.abs(y))


def load_audio(audio_list):
    WIN_LEN = 5*16000 # tamaño del audio
    
    audio_path = np.random.choice(audio_list)
    while sf.info(audio_path).duration < 5:
        audio_path = np.random.choice(audio_list)
    
    audio, fs = sf.read(audio_path)
    audio_w = audio[:WIN_LEN]
    assert fs == 16000
    
    return audio_w / np.max(np.abs(audio_w))


def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    sos = butter(order, [lowcut, highcut], fs=fs, btype='bandpass', output='sos')
    y = sosfilt(sos, data)
    
    return y


def butter_lowpass_filter(data, cutoff, fs, order):
    sos = butter(order, cutoff, fs=fs, btype='lowpass', output='sos')
    y = sosfilt(sos, data)
    
    return y


def preprocessing(data, fs, octave_band=1000):
    CUTOFF = 20 # Frecuencia de corte a 20 Hz
    
    fc = octave_band
    lowcut = fc/np.sqrt(2)
    highcut = fc*np.sqrt(2) 
    order = 4
    
    filtered_signal = butter_bandpass_filter(data, lowcut, highcut, fs, order)
    analytic_signal = hilbert(filtered_signal)
    amplitude_envelope = np.abs(analytic_signal)
    
    tae = butter_lowpass_filter(amplitude_envelope, CUTOFF, fs, order) # pasabajos a la envolvente
    tae = librosa.resample(tae, orig_sr=fs, target_sr=40) #Downsampleo
    tae = tae/np.max(tae)
    assert(tae.size==200)
    
    return tae.reshape(-1, 1)
   
    
def generate_dummy_data(audio_list, n):
    TR_MIN, TR_MAX = 0.5, 3
    
    audios = []
    labels = []
    for i in tqdm(range(n)):
        tr = np.random.choice(np.arange(TR_MIN, TR_MAX+0.1, 0.1))
        audio = load_audio(audio_list)
        rir = generate_rir(tr, 16000)
        reverb = fftconvolve(audio, rir, mode ='same')
        reverb_p = preprocessing(reverb, 16000)
        audios.append(reverb_p)
        labels.append(tr)
    
    return np.array(audios), np.array(labels)