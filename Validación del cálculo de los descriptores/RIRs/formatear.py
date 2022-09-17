import glob
from librosa import load 
from scipy.io import wavfile

palabras = glob.glob('*.wav')

for palabra in palabras:
    data, fs = load(palabra, sr=48000)

    wavfile.write(palabra, fs, data)

    print(palabra)