# Bird Sound Recognition — Dual Architecture Project (YAMNet + Google Perch)
Enfoque híbrido: Modelo de Recomendación + Modelo de Alta Precisión

### https://xeno-canto-project-docker.onrender.com/
### https://7e0d518b74a57e6258.gradio.live

Este repositorio presenta una arquitectura dual diseñada para abordar el reconocimiento automático de cantos de aves mediante dos aproximaciones complementarias:

YAMNet + XGBoost → Modelo de Recomendación
Un sistema rápido, ligero y eficiente, orientado a sugerir la especie más probable entre múltiples opciones.

Google Perch + Logistic Regression → Modelo de Alta Precisión
Un pipeline intensivo en datos, orientado a maximizar la exactitud en escenarios complejos y audios largos.

Esta dualidad no es accidental: ha sido diseñada intencionadamente para comparar arquitecturas, medir trade-offs reales y desarrollar dos soluciones complementarias con perfiles distintos de rendimiento, coste y escalabilidad.

## 1. Objetivo del Proyecto

Desarrollar un sistema completo de identificación de aves basado en audio, combinando:

Una arquitectura ligera (YAMNet) para recomendación,

Una arquitectura robusta y de alta fidelidad (Perch) para precisión,

Un dataset unificado, curado y balanceado,

Embeddings especializados para extracción de características,

Clasificadores supervisados optimizados para cada caso,

Despliegues reproducibles en entorno local y cloud.

## 2. Motivación del Enfoque Dual

El reconocimiento bioacústico presenta retos complejos:

ruido ambiental

variabilidad entre especies

micrófonos distintos

diferentes duraciones

contextos de grabación

Para capturar este espectro, diseñamos dos soluciones con roles bien definidos:

Enfoque	Modelo	Ventaja	Caso de Uso
Recomendación	YAMNet + XGBoost	Muy rápido, eficiente, excelente para top-k	Apps ligeras, móviles, filtros previos
Precisión	Google Perch + Logistic Regression	Alta fidelidad, más robusto, mejor en audios difíciles	Sistemas de análisis profesional, investigación, monitoreo

Ambos modelos comparten el dataset base, pero divergen en metodología y arquitectura, logrando:

dos pipelines complementarios

dos perfiles de inferencia

dos mecanismos de decisión

## 3. Arquitectura del Dataset

Ambos proyectos parten del mismo pipeline de ingesta.

Xeno-Canto → df_final.csv
        ↓
Curación + enriquecimiento
        ↓
Estructuración por especie
        ↓
( Rama A )                          ( Rama B )
YAMNet → 1024D embeddings           Perch → 1280D embeddings
        ↓                                   ↓
XGBoost                                Logistic Regression

Dataset global

104 especies

≥ 92.000 audios procesados

CSV enriquecido con descripciones científicas

## 4. Proyecto A — YAMNet + XGBoost (Modelo de Recomendación)
Objetivo

Proporcionar una predicción rápida y eficiente basada en embeddings YAMNet.

Características técnicas

Embedding: 1024 dimensiones

Frecuencia: 16 kHz

Clasificador: XGBoost softprob

Uso: sugerencias top-k, pipelines ágiles, despliegue en cloud con recursos limitados.

Ventajas

Ligero en memoria

Rápido en inferencia

Perfecto para sistemas de recomendación de especies probables

## 5. Proyecto B — Google Perch + Logistic Regression (Modelo de Alta Precisión)
Objetivo

Maximizar la precisión mediante embeddings especializados para bioacústica desarrollados por Google.

Características técnicas

Embedding: 1280 dimensiones

Frecuencia: 32 kHz

Ventana fija: 5 segundos

Clasificador: Softmax multinomial

Métricas alcanzadas

Top-1 Accuracy: 0.7682

Top-3 Accuracy: 0.8543

Ventajas

Muy robusto en ambientes ruidosos

Excelente en audios largos y complejos

Alta precisión para análisis profesional

## 6. Despliegue y Optimización

Cada modelo incluye mecanismos específicos de optimización:

YAMNet (Render-friendly)

Carga lazy del modelo y encoder

Uso intensivo de /tmp

Footprint < 2 GB de RAM

Tiempo de respuesta ágil

Perch (Local/Apple Silicon)

Procesamiento multi-ventana

Batch inference

Embeddings de alta fidelidad

Interfaz Gradio interactiva

## 7. Roadmap de Futuras Mejoras

Ambos modelos pueden evolucionar en paralelo:

YAMNet

LightGBM / DNN ligeras

Attention pooling

Spectrogram fusion

Perch

Entrenamiento con augmentations

Perch v2 / modelos sucesores

Softmax jerárquico taxonómico

Integración de ambos

Sistema híbrido con ensemble

Combinar precisión + rapidez

API unificada con selección automática del modelo

## 8. Conclusión

El proyecto propone dos aproximaciones complementarias al reconocimiento de aves:

Una rápida y flexible (recomendación)

Otra profunda y precisa (alta fidelidad)

Este diseño intencionado permite comparar trade-offs reales, entender mejor la dinámica bioacústica y construir un sistema diseñado para escalar en precisión, alcance y robustez.

Ambos enfoques se integran bajo un único dataset, manteniendo coherencia técnica y permitiendo evolucionar hacia una futura plataforma bioacústica completa.


