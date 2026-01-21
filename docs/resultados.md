===============================
XENO-CANTO CLASSIFICATION PROJECT
METRICS SUMMARY (XGBoost + Logistic Regression)
===============================

===========================================
MODEL 1 — XGBOOST (MULTICLASS, 104 SPECIES)
===========================================

• Features: YAMNet embeddings summary (emb_mean + emb_std + n_frames)
• Samples: ~92 577
• Classes: 104
• Train/Val/Test split: 80/10/10 (stratified)

Results:
---------
Top-1 Accuracy: ~0.41
Macro F1: ~0.40
Weighted F1: ~0.41
Top-5 Accuracy: **0.6800**

Interpretation:
---------------
XGBoost captura relaciones no lineales en el espacio de embedding (2049 dims).
El rendimiento Top-5 es adecuado para escenarios de recomendación, y comparable
a sistemas de referencia con YAMNet bajo limitaciones de CPU.

===========================================
MODEL 2 — LOGISTIC REGRESSION (SAGA)
===========================================

• Features: mismos embeddings que XGBoost
• Solver: saga (multinomial automático)
• Iteraciones: 60–80 según configuración

Results:
---------
Top-1 Accuracy: ~0.35
Macro F1: ~0.35
Weighted F1: ~0.35
Top-5 Accuracy: ~0.60–0.62 (estimado según ejecución)

Interpretation:
---------------
Modelo lineal rápido, pero limitado ante 104 clases y 2049 dimensiones.
Requiere regularización fina o reducción de dimensionalidad (PCA) para mejorar.
Útil como baseline rápido, pero inferior a XGBoost.

===========================================
CONCLUSIONES COMPARATIVAS
===========================================

• El modelo XGBoost ofrece el mejor desempeño general.
• Top-5 de 0.68 demuestra que el sistema es útil como recomendador de especies,
  especialmente en un dominio con alto solapamiento acústico.
• Logistic Regression funciona como baseline rápido de entrenamiento, pero no alcanza 
  el rendimiento requerido para producción o evaluación final.
• YAMNet + XGBoost constituye un pipeline robusto bajo restricciones de CPU y RAM.

===========================================
NOTAS TÉCNICAS
===========================================

• Dataset altamente desbalanceado entre especies.
• >92k audios y 104 clases → métrica Top-5 es más representativa que Top-1.
• Embeddings generados con YAMNet (TensorFlow Hub).
• No se usan características temporales completas (solo resumen mean+std), 
  po
