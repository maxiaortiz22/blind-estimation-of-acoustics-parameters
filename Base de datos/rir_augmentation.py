#Generación de respuestas al impulso aumentadas a partir de RIRs reales.

from decimal import DivisionByZero


if __name__ == '__main__':
    from drr_augmentation import drr_aug
    from tr_augmentation import tr_augmentation, get_audio_list
    from librosa import load
    from soundfile import write
    from numpy import arange, round, max, abs

    TR_variations = arange(1.0, 1.51, 0.01) #De 1.0 a 1.5 s con pasos de 0.01
    DRR_variations = arange(-6, 19, 1) #De -6 a 18 dB con pasos de 1


    RIRs_paths = get_audio_list('C:/Users/maxia/Desktop/C4DM Database')

    #crash_idx = RIRs_paths.index('C:/Users/maxia/Desktop/C4DM Database\X\Xx07y03.wav')

    TR_count, DRR_count, done = 0, 0, 1
    for RIR in RIRs_paths:
        data, fs = load(f'{RIR}', sr=16000)
        name = RIR.split('\\')[-1]
        name = name.split('.')[0]

        for TR_var in TR_variations:
            try:
                tr_rir = tr_augmentation(data, fs, TR_DESEADO=TR_var)
                tr_rir = tr_rir/max(abs(tr_rir))
                write(f'RIRs/{name}_{round(TR_var,1)}s.wav', tr_rir, samplerate=fs)
                TR_count+=1
            except Exception as err:
                print(f'No se pudo trabajar con el audio {RIR} a {round(TR_var,1)} s')


        for DRR_var in DRR_variations:
            try:
                drr_rir = drr_aug(data, fs,DRR_buscado=DRR_var)
                drr_rir = drr_rir/max(abs(drr_rir))
                write(f'RIRs/{name}_{DRR_var}dB.wav', drr_rir, samplerate=fs)
                DRR_count+=1
                
            except Exception as err:
                print(f'No se pudo trabajar con el audio {RIR} a {DRR_var} dB')

        print(f'Se analizaron {done}/{len(RIRs_paths)} RIRs')
        done+=1

    print(f'Se crearon {TR_count} audios de TR')
    print(f'Se crearon {DRR_count} audios de DRR')


    #drr_rir = drr_aug(data, fs=fs,DRR_buscado=5) #DRR_buscado entre -3 y 10 dB con pasos de 1 dB
    #tr_rir = tr_augmentation(data, fs, TR_DESEADO=0.3) #TR_DESEADO entre 0.2 y 3 s con pasos de 0.1

    #write('ejemplo_rir_drr.wav', drr_rir, samplerate=fs)
    #write('ejemplo_rir_tr.wav', tr_rir, samplerate=fs)