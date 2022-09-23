"""Script para probar el cálculo del TR con el método de Lundeby."""

from librosa import load
import sys
sys.path.append('../code/parameters_calculation')
from tr_lundeby import tr_lundeby, NoiseError
from filtros import BandpassFilter
import glob

if __name__ == '__main__':
    import time
    start_time = time.time()

    files = glob.glob('../data/Descartados/*.wav')
    #files = glob.glob('../data/RIRs/*.wav')

    bands = [125, 250, 500, 1000, 2000, 4000, 8000]
    type = 'octave band'
    fs = 16000
    order = 4

    bpfilter = BandpassFilter(type, fs, order, bands)

    max_ruido_dB = -45

    for file in files:
        data, fs = load(file, sr=16000) #Funciona
        filtered_audios = bpfilter.filtered_signals(data)
        print(file)

        for i, band in enumerate(bands):
            print(band)
            
            try:
                t30, _, ruidodB = tr_lundeby(filtered_audios[i], fs, max_ruido_dB)
            except (ValueError, NoiseError) as err:
                print(err.args)
                continue

            print(f'T30: {t30}')
            print(f'ruidodB: {ruidodB}')

    print("--- %s seconds ---" % (time.time() - start_time))