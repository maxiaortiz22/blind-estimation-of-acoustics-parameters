# Estimación ciega de parámetros acústicos de un recinto - 09/2022

Modelo para el entrenamiento de una red que sea capaz de 

Crear un virtual environment y correr:
pip install -r requirements.txt

Una vez instaladas las dependencias, correr el siguiente comando para evaluar los experimentos:
python run.py --config configs/exp1.py

# TODO:

* Cambiar el code/data_reader.py
* Generar un experimento
* Cambiar el modeling
* Generar un script que descarte los audios según un umbral especificado de rango para el cálculo del TR.