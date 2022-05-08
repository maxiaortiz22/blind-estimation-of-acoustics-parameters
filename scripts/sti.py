import numpy as np
from scipy.fft import fft
from filtros import bandpass_filtered_signals, butter_bandpass_filter

def find_nearest(array, value):
    '''ingresando un valor (value), busca el indice del valor mas cercano a ese valor
    en un determinado vector (array) devolviendo como resultado ese indice'''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def MTF(h,fs):
    ''' Partiendo de un vector correspondiente a una respuesta al impulso (h) y
    su frecuencia de sampleo (fs) calcula la funcion de transferencia de modulacion MTF,
    devolviendo la misma correspondiente a la parte positiva del espectro (toma un lado de
    la simetria)'''
    h2 = h**2 #Elevo al cuadrado (envelope)
    integral = np.sum(h2) #Suma energetica de todos los puntos (integral)
    hF = fft(h2) #Transformo FFT
    hF_norm = hF/integral #Normalizo por la energia total calculada
    m = abs(hF_norm) #Raiz de la suma cuadratica de parte real e imaginaria
    mtf = m[range(int((len(h2))/2))] #Me quedo con la parte positiva del espectro
    return mtf

def STI(data, fs):
    MTFS = np.zeros([7,int(len(data)/2)])
    #Creo una matriz donde cada fila es una banda de octava del impulso filtrado
    #y le calculo la MTF

    if int(fs) == 16000:
      filtered_signals = bandpass_filtered_signals(data, fs, order=4, type='octave band')

      for i in range(7):
        mtf = MTF(filtered_signals[i],fs)
        MTFS[i,:]=mtf

    elif int(fs) > 16000:
      i=0
      for F in np.array([125,250,500,1000,2000,4000,8000]):
        lowcut, highcut = F/(2**(1/2)), F*(2**(1/2))
        h_banda = butter_bandpass_filter(data, lowcut, highcut, fs, 4)
        mtf = MTF(h_banda,fs)
        MTFS[i,:]=mtf
        i+=1

    frq = np.arange(len(data))/(len(data)/fs) #paso a frecuencia
    frq = frq[range(int(len(data)/2))]

    fms = [0.63,0.8,1,1.25,1.6,2,2.5,3.15,4,5,6.3,8,10,12.5] #Frecuencias modulantes por norma

    m_indexs = np.zeros([7,14]) #Matriz de indices de modulacion. 7 bandas de octava, 14 frecs modulantes por banda

    #Obtengo los indices de modulacion para cada caso
    for j in range(7):
        i = 0
        for f in fms:
            ind = np.where(frq==find_nearest(frq,f))
            m_indexs[j,i]=MTFS[j,ind]
            i+=1

    #Calculo relacion señal a ruido aparente, acoto resultados y reescalo de 0 a 1
    SNapp = 10*np.log10(m_indexs/(1-m_indexs))
    SNapp[SNapp>15] = 15
    SNapp[SNapp<-15] = -15
    SNapp = (SNapp+15)/30

    #promedio para tener 1 valor por banda
    MTIS = np.mean(SNapp, 1)

    # Coeficientes de ponderacion - HOMBRES (Este es el que usan en el paper1!!!!!)
    alfas_H = np.array([0.085,0.127,0.230,0.233,0.309,0.224,0.173])
    betas_H = np.array([0.085,0.078,0.065,0.011,0.047,0.095]) 

    aux1 = np.sum(alfas_H*MTIS)
    aux2= np.sum(betas_H*np.sqrt(MTIS[1:]*MTIS[0:-1]))

    STI = aux1-aux2 #Sumatoria final
    
    return STI