from librosa import load
import sys
sys.path.append('../code/parameters_calculation')
from tr_lundeby import tr_lundeby
from filtros import bandpass_filtered_signals
import time
import glob
start_time = time.time()

#IMPLEMENTAR UNA FORMA DE GENERAR LOS FILTROS UNA SOLA VEZ PARA OPTIMIZAR LOS CÁLCULOS

files = glob.glob('../data/Descartados/*.wav')

bands = [125, 250, 500, 1000, 2000, 4000, 8000]
#data, fs = load(f'../data/RIRs/sintetica_Seed9756433_Tr1.5.wav', sr=16000) #Funciona
#data, fs = load(f'../data/RIRs/Xx04y00_3dB.wav', sr=16000) #Funciona
#data, fs = load(f'../data/Descartados/Wx00y01_0.2s.wav', sr=16000) #Funciona
#filtered_audios = bandpass_filtered_signals(data, fs, 4)

for file in files:
    data, fs = load(file, sr=16000) #Funciona
    filtered_audios = bandpass_filtered_signals(data, fs, 4)
    print(file)

    for i, band in enumerate(bands):
        print(band)
        
        try:
            t30 = tr_lundeby(filtered_audios[i], fs)
        except ValueError as err:
            print(err.args)
            continue

        print(t30)

print("--- %s seconds ---" % (time.time() - start_time))