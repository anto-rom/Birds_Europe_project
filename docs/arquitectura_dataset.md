# Arquitectura del Dataset — Proyectos YAMNet + XGBoost y Google Perch

Este documento describe, de forma ejecutiva, la arquitectura de datos utilizada en los dos proyectos complementarios desarrollados para la clasificación automática de cantos de aves. Ambos comparten una base común a nivel de ingesta y limpieza, pero difieren en la extracción de características y en el enfoque del modelo final.

## 1. Visión General

La arquitectura completa del dataset se articula en cinco capas funcionales:

Ingesta → Descarga de datos y metadatos desde Xeno-Canto

Curación → Limpieza, validación y enriquecimiento

Estructuración → Organización en carpetas por especie

Feature Engineering → Extracción de embeddings (YAMNet o Perch)

Modelado → Creación del dataset final para entrenamiento supervisado

Ambos proyectos siguen esta estructura, aunque difieren en la capa 4.

## 2. Arquitectura del Dataset — Proyecto YAMNet + XGBoost

Objetivo: generar un dataset balanceado de embeddings YAMNet (1024 dimensiones) para entrenar un modelo XGBoost multiclass.

### 2.1. Ingesta

Fuente: df_final.csv con metadatos: especie, URL de referencia, país, nombre común y descripción.

Descarga de audio por ID de Xeno-Canto.

### 2.2. Curación

Eliminación de audios corruptos o demasiado cortos.

Validación de URLs y duración mínima.

Incorporación de descripciones desde species_catalog_with_description.csv.

### 2.3. Estructuración
/embeddings_yamnet/
    ├── especie_1/
    ├── especie_2/
    ├── …

### 2.4. Feature Engineering

Conversión de audio → 16 kHz

Normalización

Extracción YAMNet → (N_frames, 1024)

Mean Pooling → vector final de 1024 dimensiones

Guardado en .npz (embedding + metadatos)

### 2.5. Dataset Final para Modelado

Matriz X: embeddings de tamaño (n_samples, 1024)

Vector y: etiquetas codificadas con LabelEncoder

Balance estratificado por especie

## 3. Arquitectura del Dataset — Proyecto Google Perch + Logistic Regression

Objetivo: construir un dataset masivo de embeddings Perch (1280 dimensiones) para entrenar un clasificador multinomial tipo Softmax.

### 3.1. Ingesta

Mismo dataset base: df_final.csv

Descarga del audio vía ID XC

Procesado inicial con Pydub

### 3.2. Curación

Conversión de audio a 32 kHz

Truncado/padding a 5 segundos

Validación de duración mínima

### 3.3. Estructuración
/embeddings_perch/
    ├── especie_1/
    ├── especie_2/
    ├── …

### 3.4. Feature Engineering

Extracción Perch → embedding base de 1280 dimensiones

Para embeddings con estructura temporal (T, D):
→ Mean Pooling → vector fijo (1280)

Guardado en .npz con metadata adicional (xc_id, especie)

Resultado del dataset:

104 especies

92.271 embeddings totales

### 3.5. Dataset Final para Modelado

X: matriz (n_samples, 1280)

y: especies codificadas

Normalización con StandardScaler

División estratificada 80/20

## 4. Diferencias Arquitectónicas Clave
| Componente          | Proyecto YAMNet + XGBoost | Proyecto Perch + Softmax        |
| ------------------- | ------------------------- | ------------------------------- |
| Modelo base         | YAMNet (1024D)            | Google Perch (1280D)            |
| Duración audio      | Fijo: 5s                  | Flexible                        |
| Frecuencia muestreo | 16 kHz                    | 32 kHz                          |
| Archivo final       | `xgb_model.json`          | `perch_logreg_softmax.joblib`   |
| Embeddings totales  | Menor volumen             | 92.271 embeddings               |
| Clasificador        | XGBoost Softprob          | Logistic Regression multinomial |
| Complejidad         | Alta con tuning           | Ligera y eficiente              |

Ambos proyectos comparten la misma fase de ingesta/curación, pero divergen a partir de la fase de extracción de características:

Xeno-Canto → df_final.csv
        ↓
Curación + enriquecimiento
        ↓
Estructuración por especie
        ↓
( Rama A )                        ( Rama B )
YAMNet → embedding 1024D          Perch → embedding 1280D
        ↓                                ↓
XGBoost                              Logistic Regression
        ↓                                ↓
Modelo final                        Modelo final



