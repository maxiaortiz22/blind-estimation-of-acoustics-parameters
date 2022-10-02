# Estimación ciega de parámetros acústicos de un recinto - 09/2022

Tesis de grado para optar por el título de Ingeniero de Sonido.

En este repositorio se encuentra el código utilizado para la generación de la base de datos y el entrenamiento de un modelo para la estimación ciega de parámetros acústicos de un recinto.

En particular, esta red es capaz de estimar los parámetros T30, C50, C80 y D50 de un audio de voz reverberado.

---

Crear un virtual environment y correr:
pip install -r requirements.txt

Una vez instaladas las dependencias, correr el siguiente comando para evaluar los experimentos:
python run.py --config configs/exp1.py

---

# TODO:

* Cambiar el code/data_reader.py
* Generar un experimento
* Cambiar el modeling
* Generar un script que descarte los audios según un umbral especificado de rango para el cálculo del TR.