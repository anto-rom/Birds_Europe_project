#  Hoja de Ruta del Proyecto — Bird Sound Classifier (YAMNet + XGBoost) MODELO DE RECOMENDACIÓN
### https://xeno-canto-project-docker.onrender.com/

## 1. Definición del alcance

Objetivo principal:
Desarrollar un clasificador funcional de cantos de aves basado en embeddings YAMNet y un modelo supervisado (XGBoost), con despliegue web operativo y optimizado para ejecución cloud con recursos limitados.

Fuente: https://xeno-canto.org/

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

#  Hoja de Ruta del Proyecto — Bird Sound Recognition (Google Perch + Tensorflow) MODELO DE PRECISIÓN gabrielajara2982/Proyecto_Xeno_Canto_Classifier_Google_Perch
(Anexo técnico basado en Google Perch y embeddings de 1280 dimensiones)

### https://proyecto-xeno-canto-classifier-google.onrender.com/

## 1. Objetivo del Proyecto

El propósito principal del proyecto es desarrollar un sistema robusto de reconocimiento de cantos de aves utilizando Google Perch, un modelo de representación de audio entrenado por Google para bioacústica avanzada.
El pipeline integra:

Descarga masiva de audios desde Xeno-Canto

Procesado y limpieza

Generación de embeddings con Perch

Entrenamiento de un clasificador supervisado

Obtención de métricas de precisión

Validación funcional mediante una interfaz interactiva

## 2. Preparación del Dataset
### 2.1. Ingesta de datos

Lectura del CSV final procedente del repositorio principal (df_final.csv)

Estructura base del dataset:

scientificName

references

vernacularName

country

description

##2.2. Estructuración por especie

Creación automática de carpetas por especie.

Organización de todos los audios procesados en subdirectorios.

## 3. Pipeline Google Perch
### 3.1. Descarga y carga del modelo

Descarga desde Kaggle Models:
bird-vocalization-classifier

Alternativa de respaldo desde TensorFlow Hub si falla la anterior.

Uso de model.signatures['serving_default'] para obtener el endpoint inferencial.

3.2. Arquitectura Perch

Modelo especializado en bioacústica con vector de salida fijo de 1280 dimensiones.

Procesamiento interno basado en análisis tiempo-frecuencia de alta resolución.

## 4. Generación de Embeddings
### 4.1. Procesamiento de audio

Descarga del audio original mediante URL directa de Xeno-Canto.

Conversión con Pydub:

Mono

32 kHz

Ventana fija de 5 segundos

Relleno o truncado según duración original

### 4.2. Extracción del embedding

Uso de Perch.process_array(samples)

Obtención del vector de 1280 valores para cada grabación

Guardado en formato npz con:

embedding

species

xc_id

### 4.3. Stats del dataset procesado

Total especies: 104

Total embeddings generados: 92.271

## 5. Entrenamiento del Modelo Supervisado
### 5.1. Construcción del dataset final

Carga del conjunto completo de embeddings.

Aplicación de mean pooling temporal cuando el embedding llega en formato (T, D).

Ensamblado en matrices:

X → embeddings

y → especies

5.2. Preprocesado

Codificación de labels con LabelEncoder.

Normalización con StandardScaler.

### 5.3. Entrenamiento del modelo

Algoritmo: Regresión logística multinomial (Softmax)

Solver: lbfgs

Iteraciones: 500

Divisions: 80% train | 20% test

Stratify por especie

### 5.4. Métricas finales de precisión

Top-1 Accuracy: 0.7682

Top-3 Accuracy: 0.8543

5.5. Artefactos generados

perch_logreg_softmax.joblib

label_encoder.joblib

## 6. Validación y Testing
### 6.1. Interfaz de pruebas

Implementación de un clasificador funcional sobre Gradio.

Capacidad de cargar MP3/WAV y devolver especie + confianza.

Sistema multiplataforma optimizado para Apple Silicon (M2).

### 6.2. Features clave de la inferencia

Procesado en ventanas de 5 segundos para audios largos.

Media de todas las ventanas para generar un vector robusto.

Umbral de confianza aplicado para evitar falsas predicciones (<40%).

## 7. Resultados y Conclusiones

El modelo demuestra un rendimiento sólido, especialmente tratándose de audio silvestre con alta variabilidad.

Perch ofrece embeddings muy informativos (1280D), lo que facilita un clasificador simple pero efectivo.

El pipeline está limpio, modular y escalable.

La metodología es óptima para aplicaciones de campo, proyectos educativos y prototipos de investigación.

## 8. Roadmap Evolutivo

Incorporación de Transformers de audio o modelos de espectrogramas.

Fine-tuning ligero sobre Perch mediante frameworks como audiocraft o kapre.

Ampliación del dataset a otras regiones (Latam, Asia).

Integración en una API servida vía Flask o FastAPI.

Dashboard web para visualización de predicciones en tiempo real.

