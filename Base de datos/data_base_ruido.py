if __name__ == '__main__':
    from os import listdir
    from soundfile import write
    from librosa import load
    import numpy as np
    from random import randrange
    from scipy.signal import fftconvolve
    import snr_calculator

    audio_path = listdir('ACE Challenge selected/')
    rir_path = listdir('RIRs/')
    noise_path = listdir('Ruidos/')

    COUNT = 1
    tot_audios = len(audio_path)*len(rir_path)
    for audio in audio_path:
        audio_data, fs = load(f'ACE Challenge selected/{audio}', sr=16000, duration=5.0)
        audio_name = audio.split('.wav')[0]

        for rir in rir_path:
            #Tomo una rir
            rir_data, _ = load(f'RIRs/{rir}', sr=16000)
            rir_name = rir.split('.wav')[0]

            #Tomo un ruido cualquiera:
            idx_noise = randrange(len(noise_path))
            noise_data, _ = load(f'Ruidos/{noise_path[idx_noise]}', sr=16000, duration=5.0)
            noise_name = noise_path[idx_noise].split('.wav')[0]

            reverbed_audio = fftconvolve(audio_data, rir_data, mode='same')

            reverbed_audio = reverbed_audio / np.max(np.abs(reverbed_audio))

            #Agrego ruido para tener SNR entre -20 y -5 dB:
            rms_signal = snr_calculator.rms(reverbed_audio)
            rms_noise = snr_calculator.rms(noise_data)

            snr_required = np.random.uniform(-5, 20, 1)[0]

            comp = snr_calculator.rms_comp(rms_signal, rms_noise, snr_required)

            noise_data_comp = noise_data*comp

            reverbed_noisy_audio = reverbed_audio + noise_data_comp
            reverbed_noisy_audio = reverbed_noisy_audio / np.max(np.abs(reverbed_noisy_audio))


            snr_required = np.round(snr_required, 3)
            write(f'ReverbedAudios_Ruido/{audio_name}--{noise_name}_{snr_required}dB--{rir_name}.wav',
                  reverbed_noisy_audio, fs)
            
            print(f'Se crearon {COUNT} audios de {tot_audios}')
            COUNT+=1