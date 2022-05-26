from definition import definition
from clarity import clarity
from center_time import center_time
from tr_lundeby import tr_lundeby
from filtros import bandpass_filtered_signals
import numpy as np

def descriptors(data, fs):
    
    #filtered_data = bandpass_filtered_signals(data, fs, order=4, type='octave band')
    #bandas = [125, 250, 500, 1000, 2000, 4000, 8000]
    #T30: 
    TRs = tr_lundeby(data, fs)
    T30 = TRs[-1]

    #C80:
    C80 = clarity(80, data, fs)

    #D50:
    D50 = definition(data, fs)

    #Ts:
    Ts = center_time(data, fs)

    return [T30, C80, D50, Ts]

#ejemplo_rir:
#T30:0,518863	0,401405	0,397246	0,385379	0,353634	0,314426	0,290911		
#C80:10,047853	15,849098	10,133373	13,933165	16,535813	16,983064	20,030656		
#D50: 80,938435	88,908590	73,308220	88,496284	90,834107	91,773721	94,469815		
#Ts: 175,979566	91,458671	65,806218	42,063388	30,133390	23,406678	16,601593		

#ACE audio:
#T30: 0,843719	1,275019	1,484793	1,624404	1,615517	1,322724	0,830876	
#C80: 5,817376	6,609946	5,873018	6,022028	8,434563	10,606210	20,203140	
#D50: 71,093936	72,292855	71,578766	75,024595	84,509075	88,911379	98,461925		
#Ts: 167,001923	104,239032	82,496910	62,493177	38,071223	23,079823	4,491612	
	
#ACE 2 audio:
#T30: 0,773742	0,894843	0,654813	0,632261	0,852172	0,801094	0,666497		
#C80: 4,607238	6,062472	9,766924	12,544614	12,454839	12,959086	18,641926		
#D50: 21,478184	71,826385	82,292959	90,438024	89,180450	90,287680	97,276377			
#Ts: 212,925802	115,048737	59,668010	34,972927	26,290105	20,529610	7,537185	  	


if __name__ == '__main__':
    from librosa import load
    import os

    os.system('cls') 

    data, fs = load('ejemplo_rir.wav', sr=16000)

    in_max = np.where(abs(data) == np.max(abs(data)))[0]  
    in_max = int(in_max[0])
    signal = data[(in_max):]

    filtered_data = bandpass_filtered_signals(signal, fs, order=4, type='octave band')
    bandas = [125, 250, 500, 1000, 2000, 4000, 8000]

    for i, banda in enumerate(bandas):
        descriptors_results = descriptors(filtered_data[i], fs)
        print(f'Para la banda {banda}, los resultados fueron: {descriptors_results}')
    
    #for i, banda in enumerate(bandas):
    #    Ts = center_time(bandpass_filtered_signals(data, fs, order=4, type='octave band')[i], fs)
    #    print(f'Para la banda {banda}, los resultados de Ts fueron: {Ts}')