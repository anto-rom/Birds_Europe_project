from pathlib import Path
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Ruta del dataset REAL
DATASET = Path(r"C:\Projects\Xeno_Canto_Project_CLEAN\data\raw\species_catalog_with_description.csv")

# Ruta de salida dentro del repo CLEAN
OUT = Path(r"C:\Projects\Xeno_Canto_Project_CLEAN\src\label_encoder.joblib")

df = pd.read_csv(DATASET)

label_col = "scientificName"

if label_col not in df.columns:
    raise ValueError(
        f"ERROR: La columna '{label_col}' no existe en el CSV. "
        f"Columnas disponibles: {df.columns.tolist()}"
    )

labels = (
    df[label_col]
    .dropna()
    .astype(str)
    .str.strip()
)

# Quitar vacíos si hubiera
labels = labels[labels.ne("")]

# Crear encoder consistente
le = LabelEncoder()
le.fit(labels.unique())

OUT.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(le, OUT)

print("\nDONE")
print("Encoder guardado en:", OUT)
print("Número de clases:", len(le.classes_))
print("Primeras clases:", le.classes_[:10])
