## 🐦 Reconocimento de aves por su canto – Google PERCH (Apple Silicon M2)

Identificación automática de cantos de aves usando Google PERCH y regresión logística multinomial Softmax.

Este repositorio contiene un modelo entrenado en Mac M2, junto con un script Python (app.py) que permite ejecutar la identificación de especies en audio mediante Gradio con enlace público temporal.

## 📦 Contenido del repositorio


├── app_perch.py                     ← Script Python para ejecutar la app
├── perch_logreg_softmax.joblib     ← Clasificador softmax entrenado
├── label_encoder.joblib            ← Label Encoder de especies
├── requirements.txt                ← Librerías necesarias
├── Google_perch.ipynb              ← Codigo

## 🖥️ Requisitos

Sistema operativo: macOS (Apple Silicon M1/M2 recomendado)

Python: 3.10.x

Conda (opcional pero recomendado)

Dependencias (según requirements.txt)

## 🐦 Cómo funciona

El audio se carga y se normaliza a 32 kHz mono.

Se divide en ventanas de 5 segundos (160.000 muestras) para análisis multi-ventana.

Cada ventana se pasa al modelo PERCH para generar embeddings de 1280 dimensiones.

Se promedia el embedding de todas las ventanas.

Se pasa el embedding al clasificador softmax (.joblib) entrenado en Mac M2.

Se devuelve:

La especie predicha

Confianza (%) de la predicción

Tiempo estimado de análisis en CPU M2

## 🔑 Notas importantes

El modelo está entrenado en Mac M2 (ARM).

No es directamente compatible con Linux x86_64. Para deploy en la nube (Render, Heroku, Docker) es necesario reentrenar o reconstruir el .joblib en Linux.

El enlace público Gradio generado con share=True es temporal y dura mientras tu Mac esté encendida y el script corriendo.

Ideal para demos, pruebas o compartir con colegas.

## 📌 Futuras mejoras

Guardar historial de predicciones

Integración con Streamlit para un hosting más permanente 

Optimización de tiempo de CPU para audios largos

Entrenamiento con todas las especies de aves del mundo con el fin de consolidad una enciclopedia incluso incluyendo con entrenamiento de imagenes e informacion de otras especies del reino animal con el  fin de consulta academica a nivel global. 