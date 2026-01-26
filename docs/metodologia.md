#Hoja de Ruta del Proyecto — Bird Sound Classifier (YAMNet + XGBoost)


## 1. Definición del alcance

Objetivo principal:
Desarrollar un clasificador funcional de cantos de aves basado en embeddings YAMNet y un modelo supervisado (XGBoost), con despliegue web operativo y optimizado para ejecución cloud con recursos limitados.

Outputs esperados:

Dataset balanceado y curado de aves.

Embeddings generados con YAMNet.

Modelo de Machine Learning entrenado.

API Flask funcional.

Despliegue en Render con footprint < 2 GB.

Documentación completa del pipeline.

## 2. Construcción del dataset

Tareas ejecutadas:

Descarga de audios desde Xeno-Canto mediante API.

Limpieza inicial (duplicados, audios corruptos, registros sin referencias).

Enriquecimiento con descripciones a partir de species_catalog_with_description.csv.

Balanceo del dataset:

Muestreo estratificado por especie.

Número máximo por especie (CAP).

Exportación final del dataset preparado.

Entregables:

df_final.csv

df_balanced.csv

CSV con descripciones enriquecidas

## 3. Generación de embeddings YAMNet

Acciones clave:

Configurar pipeline eficiente para lectura de cada audio.

Conversión a 16 kHz para garantizar compatibilidad.

Extracción de embeddings YAMNet.

Guardado en estructura organizada por especie.

Reducción de memoria: procesamiento iterativo sin cargar todo en RAM.

Entregables:

Carpeta embeddings_yamnet/

Notebook de generación y validación

## 4. Entrenamiento del modelo supervisado

Decisiones técnicas:

Comparativa inicial de algoritmos (LR, RF, XGB).

Selección final: XGBoost por mejor equilibrio entre precisión y rendimiento.

Preprocesado:

Normalización

Label encoding

Validación cruzada y métricas finales.

Entregables:

xgb_model.json

label_encoder.joblib

Notebook de entrenamiento con análisis de métricas

## 5. Desarrollo de la API / Interfaz Flask

Objetivos del backend:

Cargar YAMNet una sola vez.

Cargar el XGBoost y el encoder de forma eficiente.

Permitir upload de audio + predicción + descripción.

Plantilla HTML ligera y responsive.

Optimización clave:

Lazy load del modelo.

Eliminación de cargas masivas de CSV en memoria.

Control de buffers y GC después de cada inferencia.

Entregables:

prototype.py

Plantillas templates/

Archivos estáticos static/

## 6. Optimización para ejecutar en Render (2 GB RAM)

Acciones realizadas:

Descarga de modelo y encoder en /tmp.

Eliminación de datasets innecesarios en el arranque.

Compresión del modelo en un ZIP único en GitHub Releases.

Reducción del uso de pandas a momentos puntuales.

Control del cache YAMNet con TFHUB_CACHE_DIR=/tmp.

Revisión y limpieza de variables temporales.

Resultado: App funcionando de manera estable sin memory leaks.

## 7. Despliegue

Pasos seguidos:

Release en GitHub con el ZIP del modelo.

Configuración del servicio Render:

Starter plan con 2 GB.

Gunicorn no requerido (solo Flask).

Validación de logs y requests.

Fix de errores 502 relacionados con timeouts y rutas.

Test end-to-end con audios reales.

Entregable:

URL pública de la app.

## 8. Documentación y mantenimiento

Incluye:

README técnico del repositorio.

Instrucciones para regenerar el dataset.

Manual para futuras actualizaciones del modelo.

Guía de troubleshooting (errores comunes de Render, rutas, permisos, dependencias).

Roadmap evolutivo: versión 2.0 con Streamlit, base de datos, cola de procesos o CDN.

## 9. Próximos pasos recomendados (roadmap evolutivo)

Front-end más interactivo (Streamlit o React).

Batch processing para audios largos.

Modelo 2.0 con LightGBM o Transformer pequeño.

Logging estructurado + monitorización básica.

Cache de predicciones frecuentes.

Dockerización para despliegues multi-cloud.

Incorporación de spectrogramas como features extras.


