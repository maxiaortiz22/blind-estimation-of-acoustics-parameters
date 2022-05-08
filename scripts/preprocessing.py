from librosa import load
from tae import TAE
from numpy import round

def preprocessing(files_name, band):
    tae = [[]] * int(len(files_name)) #Genero una lista de listas vacías
    T30 = [] #Genero una lista vacía

    i=0
    for file in files_name:
        data, fs = load(file, sr=16000)

        tae[i] = TAE(data, int(fs), band) #Calculo el TAE en la banda especificada
        tae[i] = tae[i].reshape(-1, 1) #Le agrego una dimensión más para que lo tome tensorflow

        Tr = file.split('.wav')[0]
        Tr = round(float(Tr.split('Tr')[-1]), 1)
        T30.append( Tr ) #Cargo el T30 del audio
        
        i+=1
        
    return tae, T30