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
    import pickle

    audio_path = listdir('RIRs/')
    audio_path = [signal.split('.wav')[0] for signal in audio_path]

    bandas = [125, 250, 500, 1000, 2000, 4000, 8000]

    print('Inicio la copia de parámetros:')
    
    for band in bandas:
        with open(f'Parametros/Parametros_{str(band)}.pickle', 'rb') as handle:
                        params_rirs = pickle.load(handle)

        params_keys = params_rirs.keys()
        signals_params = {}
        for signal in audio_path:
            #print(param_key)
            for param_key in params_keys:
                if signal in param_key:
                    #print(param_key)
                    signals_params[signal] = params_rirs[param_key]
                    break

        with open(f'Parametros_Ruido/Parametros_aux_{str(band)}.pickle', 'wb') as handle:
                    pickle.dump(signals_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(f'\nLista la banda {band}. Con {len(signals_params.keys())} audios.')
    
    audio_path = listdir('RIRs/')
    
    signals_path = listdir('ReverbedAudios_Ruido/')
    signals_path = [signal.split('.wav')[0] for signal in signals_path]

    print('Iniciando copia a los respectivos diccionarios:')
    
    for band in bandas:
        with open(f'Parametros_Ruido/Parametros_aux_{str(band)}.pickle', 'rb') as handle:
                        params_rirs = pickle.load(handle)

        params_keys = params_rirs.keys()
        signals_params = {}
        for param_key in params_keys:
            #print(param_key)
            for signal in signals_path:
                if param_key in signal:
                    #print(param_key)
                    signals_params[signal] = params_rirs[param_key]

        with open(f'Parametros_Ruido/Parametros_{str(band)}.pickle', 'wb') as handle:
                    pickle.dump(signals_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(f'\nLista la banda {band}. Con {len(signals_params.keys())} audios.')