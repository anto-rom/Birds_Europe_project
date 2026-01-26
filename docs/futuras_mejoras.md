# Diagramas Mermaid — Roadmap de Escalabilidad
## 1. Visión General de la Evolución de Ambos Modelos

flowchart TD
    A[Dataset Xeno-Canto<br>df_final.csv] --> B[Curación y Limpieza]
    B --> C[Estructuración por especie]

    C --> D1[YAMNet Embeddings<br>1024D]
    C --> D2[Google Perch Embeddings<br>1280D]

    D1 --> E1[XGBoost Softprob]
    D2 --> E2[Logistic Regression<br>Softmax]

    E1 --> F1[Versión Actual YAMNet]
    E2 --> F2[Versión Actual Perch]

    F1 --> G1[Mejoras Futuras YAMNet<br>- LightGBM<br>- Attention Pooling<br>- Transfer Learning]
    F2 --> G2[Mejoras Futuras Perch<br>- Batch Processing<br>- Augmentation<br>- Redes Densas]

    G1 --> H[Arquitectura Escalable en Cloud]
    G2 --> H
## 2. Roadmap de Escalabilidad — Vista Global
mindmap
  root((Roadmap de Escalabilidad))

    Dataset
      Expansión geográfica
      Más fuentes (eBird, ML)
      Control automático de calidad
      Pipeline incremental

    Feature Engineering
      YAMNet
        Attention Pooling
        Fusión de espectrogramas
        Fine-tuning
      Perch
        Batch processing
        Augmentations
        Perch v2 / modelos sucesores

    Modelado
      XGBoost → LightGBM / CatBoost
      Logistic Regression → DNN
      Ensembles YAMNet + Perch
      Softmax jerárquico

    Infraestructura
      Docker multi-stage
      AWS / Azure / Render
      Serverless
      Cache distribuida

    MLOps
      Retraining automático
      Validación continua
      Monitoreo de latencia
      Trazabilidad de modelos

    UX / API
      Streamlit / React UI
      API REST pública
      Top-3 predictions
      Visualización espectrogramas

## 3. Arquitectura Integrada Actual → Futura

sequenceDiagram
    participant User
    participant API as API Modelo
    participant FE as Extracción de Embeddings
    participant ML as Modelo Supervisado
    participant Future as Infraestructura Escalable

    User->>API: Subida de Audio
    API->>FE: Procesar audio (YAMNet/Perch)
    FE->>ML: Embedding → vector 1024/1280D
    ML->>API: Predicción + Confianza
    API->>User: Resultado final

    Note over Future: Futuras mejoras<br>• Contenedores<br>• Serverless<br>• Ensembles<br>• Dataset global<br>• Monitorización
## 4. Diagrama de Componentes — Escalabilidad

graph LR
    subgraph Ingesta
        A[Descarga Xeno-Canto]
        B[Curación + Enriquecimiento]
        C[Estructuración por especie]
    end

    subgraph Features
        D1[YAMNet 1024D]
        D2[Perch 1280D]
    end

    subgraph Modelos
        E1[XGBoost]
        E2[Logistic Regression]
        E3[Modelos futuros<br>DNN / Transformers]
    end

    subgraph Infraestructura
        F1[Docker]
        F2[API REST]
        F3[Render / AWS / Azure]
        F4[Monitorización]
    end

    A-->B-->C-->D1
    C-->D2
    D1-->E1
    D2-->E2
    E1-->E3
    E2-->E3
    E3-->F1-->F2-->F3-->F4

