from librosa import load
import numpy as np
from tr_lundeby import tr_lundeby
from tr_convencional import tr_convencional
from filtros import bandpass_filtered_signals
from band_ploter import band_ploter

data, fs = load('ejemplo_rir.wav', sr=16000)

filtered_data = bandpass_filtered_signals(data, fs, order=4, type='octave band')

bandas = [125, 250, 500, 1000, 2000, 4000, 8000]

t30_lundeby, t30_convencional = [], []
for i in range(len(bandas)):
    t30_lundeby.append(tr_lundeby(filtered_data[i], fs)[-1])
    t30_convencional.append(tr_convencional(filtered_data[i], fs, rt='t30'))

print(f'T30 con Lundeby:')
print(t30_lundeby)

print(f'T30 convencional:')
print(t30_convencional)

t30_lundeby = np.array(t30_lundeby)
t30_convencional = np.array(t30_convencional)

print('Diferencia:')
dif = t30_lundeby-t30_convencional
print(dif)

band_ploter(t30_lundeby, 'Tr usando Lundeby')
band_ploter(t30_convencional, 'Tr convencional')