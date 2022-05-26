#Generación de respuestas al impulso aumentadas a partir de RIRs reales.

if __name__ == '__main__':
    from drr_augmentation import drr_aug
    from tr_augmentation import tr_augmentation
    from librosa import load
    from soundfile import write

    data, fs = load('ejemplo_rir.wav', sr=16000)

    drr_rir = drr_aug(data, fs=fs,DRR_buscado=5) #DRR_buscado entre -3 y 10 dB con pasos de 1 dB
    tr_rir = tr_augmentation(data, fs, TR_DESEADO=0.3) #TR_DESEADO entre 0.1 y 1.2 s con pasos de 0.05

    write('ejemplo_rir_drr.wav', drr_rir, samplerate=fs)
    write('ejemplo_rir_tr.wav', tr_rir, samplerate=fs)