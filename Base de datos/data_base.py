if __name__ == '__main__':
    from os import listdir
    from soundfile import write
    from librosa import load
    import numpy as np
    from scipy.signal import fftconvolve

    audio_path = listdir('ACE Challenge selected/')
    rir_path = listdir('RIRs/')

    COUNT = 1
    tot_audios = len(audio_path)*len(rir_path)
    for audio in audio_path:
        audio_data, fs = load(f'ACE Challenge selected/{audio}', sr=16000, duration=5.0)
        audio_name = audio.split('.wav')[0]

        for rir in rir_path:
            rir_data, _ = load(f'RIRs/{rir}', sr=16000)
            rir_name = rir.split('.wav')[0]

            reverbed_audio = fftconvolve(audio_data, rir_data, mode='same')

            reverbed_audio = reverbed_audio / np.max(np.abs(reverbed_audio))

            write(f'ReverbedAudios/{audio_name}-{rir_name}.wav',
                  reverbed_audio, fs)
            
            print(f'Se crearon {COUNT} audios de {tot_audios}')
            COUNT+=1