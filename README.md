# Estimación ciega de parámetros acústicos de un recinto - 10/2022

Tesis de grado para optar por el título de Ingeniero de Sonido.

En este repositorio se encuentra el código utilizado para la generación de la base de datos y el entrenamiento de un modelo para la estimación ciega de parámetros acústicos de un recinto.

En particular, esta red es capaz de estimar los parámetros T30, C50, C80 y D50 de un audio de voz reverberado.

---
Este código se realizó en el sistema operativo Windows con python 3.9.13. Los módulos y sus versiones utilizadas se encuentran en el archivo *requirements.txt*, las cuales pueden variar según el sistema operativo.

Crear un virtual environment y correr:
pip install -r requirements.txt

Una vez instaladas las dependencias, correr el siguiente comando para evaluar los experimentos:
python run.py --config configs/exp1.py
