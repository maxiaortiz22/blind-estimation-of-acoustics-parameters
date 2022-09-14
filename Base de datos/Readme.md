# Generación de base de datos

Teniendo una selección de audios de habla y una base de datos de respuesta al impulso, con esta serie de scripts es posible calcular los TAE (Temporal Amplitud Envelopes) y los descriptores de los audios reverberados.

Para esto, primero se realiza una aumentación de las RIRs de nuestra base de datos (eligiendo límites de TR y de DRR) y, además, se generan respuestas al impulso sintéticas.

Una vez obtenido nuestro Dataset de RIRs, se las convoluciona con los audios de voz seleccionados en la carpeta *ACE Challenge selected* (los cuales provienen de la base de datos del ACE Challenge) y se los guarda en la carpeta *ReverbedAudios* o *ReverbedAudios_Ruido* según sea el caso.

La diferencia entre ambas carpetas anteriormente descriptas es que a los audios reverberados de la segunda se les agrega ruido de forma tal que se obtenga una SNR entre -5 y 20 dB.

Una vez obtenidos los audios reverberados, el siguiente paso es encontrar sus valores de TAE y los valores de los descriptores de cada RIR de nuestra base de datos aumentada.