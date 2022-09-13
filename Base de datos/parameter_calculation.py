class DescriptorError(Exception):
    """Defino un tipo de error en el cálculo de los descriptores"""
    """Exception raised for errors in the input descriptor.

    Attributes:
        descriptor_value -- value which caused the error
        message -- explanation of the error
    """

    def __init__(self, descriptor_value, message):
        self.descriptor_value = descriptor_value
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.message} {self.descriptor_value}. Se decarta el audio.'

if __name__ == '__main__':
    from os import listdir
    from librosa import load
    from tr_lundeby import tr_lundeby
    #from center_time import center_time
    from clarity import clarity
    from definition import definition
    import pickle
    from math import isnan, isinf
    from filtros import bandpass_filtered_signals
    from shutil import move

    audio_path = listdir('RIRs/')

    voice_name = listdir('ACE Challenge selected')
    voice_name = [name.split('.wav')[0] for name in voice_name]

    bandas = [125, 250, 500, 1000, 2000, 4000, 8000]

    dic = {}
    with open(f'Parametros/Parametros_125.pickle', 'wb') as handle:
        pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'Parametros/Parametros_250.pickle', 'wb') as handle:
        pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'Parametros/Parametros_500.pickle', 'wb') as handle:
        pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'Parametros/Parametros_1000.pickle', 'wb') as handle:
        pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'Parametros/Parametros_2000.pickle', 'wb') as handle:
        pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'Parametros/Parametros_4000.pickle', 'wb') as handle:
        pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'Parametros/Parametros_8000.pickle', 'wb') as handle:
        pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    COUNT=1
    tot = len(audio_path)*7
    for audio in audio_path:
        data, fs = load(f'RIRs/{audio}', sr=16000)
        audio_name = audio.split('.wav')[0]

        filtered_audios = bandpass_filtered_signals(data, fs, 4)

        t30_list = []
        c50_list = []
        c80_list = []
        d50_list = []

        try:

            for i, band in enumerate(bandas):

                t30 = tr_lundeby(filtered_audios[i], fs)
                t30 = t30[-1]

                c50 = clarity(50, filtered_audios[i], fs)
                
                c80 = clarity(80, filtered_audios[i], fs)

                d50 = definition(filtered_audios[i], fs)

                if isnan(t30) | isinf(t30):
                    raise DescriptorError(descriptor_value=t30, message='El T30 dio:')
                if isnan(c50) | isinf(c50):
                    raise DescriptorError(descriptor_value=c50, message='El C50 dio:')
                if isnan(c80) | isinf(c80):
                    raise DescriptorError(descriptor_value=c80, message='El C80 dio:')
                if isnan(d50) | isinf(d50):
                    raise DescriptorError(descriptor_value=d50 ,message='El D50 dio:')

                t30_list.append(t30)
                c50_list.append(c50)
                c80_list.append(c80)
                d50_list.append(d50)


            check_t30 = [True for i in range(len(t30_list)) if t30_list[i] > t30_list[3]+4] #Uso el valor en 1000 como referencia
            
            if any(check_t30): #Descarto el audio si en alguna banda hay una diferencia mayor a 4 segundos respecto a 1 kHz
                source_path = f'RIRs/{audio}'
                destination_path = f'Descartadas/{audio}'

                move(source_path, destination_path)

                COUNT+=7 #Para no perder la cuenta del total le sumo 7 como su hubiese pasado!
            
            else:
                for i, band in enumerate(bandas):
                    with open(f'Parametros/Parametros_{str(band)}.pickle', 'rb') as handle:
                        param_dic = pickle.load(handle)

                    
                    for voice in voice_name:
                        param_dic[f'{voice}-{audio_name}'] = [t30_list[i], c50_list[i], c80_list[i], d50_list[i]]

                    print(f'Se calcularon {COUNT}/{tot} audios: banda actual {band}, audio {audio_name}')
                    COUNT+=1
            
                    with open(f'Parametros/Parametros_{str(band)}.pickle', 'wb') as handle:
                        pickle.dump(param_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        except Exception as err:

            print(repr(err))

            source_path = f'RIRs/{audio}'
            destination_path = f'Descartadas/{audio}'

            move(source_path, destination_path)

            COUNT+=7 #Para no perder la cuenta del total le sumo 7 como su hubiese pasado!
