import os
from pathlib import Path

# Cache cross-platform (Render/Linux usa /tmp)
os.environ["TFHUB_CACHE_DIR"] = os.getenv(
    "TFHUB_CACHE_DIR",
    str(Path("/tmp") / "tfhub_cache")
)

from flask import Flask, request, render_template
import numpy as np
import joblib
import pandas as pd
import tensorflow_hub as hub
import tensorflow as tf
import librosa
import requests
from xgboost import XGBClassifier

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

XGB_MODEL_JSON_PATH = Path(os.getenv("XGB_MODEL_PATH", str(Path("/tmp") / "xgb_model.json")))
ENCODER_PATH        = BASE_DIR / "label_encoder.joblib"

GITHUB_OWNER = "anto-rom"
GITHUB_REPO  = "Xeno_Canto_Project"
RELEASE_ASSET_NAME = "xgb_model.json"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# ---- Cargar YAMNet con IO device fijado ----
load_opts = tf.saved_model.LoadOptions(experimental_io_device="/job:localhost")
YAMNET_MODEL = hub.load("https://tfhub.dev/google/yamnet/1", options=load_opts)

TARGET_SR = 16000


# -----------------------------
# DOWNLOAD HELPERS
# -----------------------------
def _github_headers():
    h = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return h


def download_release_asset_if_missing(local_path: Path, asset_name: str):
    if local_path.exists() and local_path.stat().st_size > 0:
        return

    # Descarga directa del asset del último release (sin API -> sin rate limit)
    download_url = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest/download/{asset_name}"

    local_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(download_url, stream=True, timeout=120, allow_redirects=True) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    if not local_path.exists() or local_path.stat().st_size == 0:
        raise RuntimeError("Descarga completada pero el archivo quedó vacío/corrupto.")



# -----------------------------
# BLINDED LOADERS
# -----------------------------
def resolve_description_file(project_root: Path) -> Path:
    """
    Busca el archivo de descripciones en rutas típicas y soporta XLSX/CSV.
    Prioriza el XLSX que nos dijiste.
    """
    candidates = [
        project_root / "data" / "raw" / "scientificName_description.xlsx",
        project_root / "data" / "raw" / "species_catalog_with_description.xlsx",
        project_root / "data" / "processed" / "species_catalog_with_description.xlsx",
        project_root / "data" / "processed" / "species_catalog_with_description.csv",
        project_root / "src" / "species_catalog_with_description.csv",
    ]

    for p in candidates:
        if p.exists() and p.is_file() and p.stat().st_size > 0:
            return p

    # mensaje con “rutas esperadas” para troubleshooting
    expected = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "No encuentro ningún archivo de descripciones. He buscado en:\n"
        f"{expected}\n"
        "Acción: coloca el fichero en una de esas rutas o actualiza la lista de candidatos."
    )


def load_descriptions(file_path: Path) -> dict:
    """
    Carga CSV o XLSX y devuelve dict {scientificName: description}.
    Valida columnas y sanea strings.
    """
    suffix = file_path.suffix.lower()

    if suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    elif suffix == ".csv":
        # robusto: intenta separadores comunes
        try:
            df = pd.read_csv(file_path)
        except Exception:
            df = pd.read_csv(file_path, sep=";")
    else:
        raise ValueError(f"Formato no soportado: {suffix}. Usa .xlsx/.xls/.csv")

    # Normaliza nombres de columnas (por si vienen con espacios raros)
    df.columns = [str(c).strip() for c in df.columns]

    required = {"scientificName", "description"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"El archivo {file_path.name} no tiene las columnas requeridas: {sorted(missing)}. "
            f"Columnas encontradas: {list(df.columns)}"
        )

    # Limpieza básica
    df = df[["scientificName", "description"]].copy()
    df["scientificName"] = df["scientificName"].astype(str).str.strip()
    df["description"] = df["description"].astype(str).str.strip()

    # Filtra filas inválidas
    df = df[df["scientificName"].ne("") & df["scientificName"].ne("nan")]
    df = df[df["description"].ne("") & df["description"].ne("nan")]

    # Si hay duplicados, nos quedamos con el último (criterio simple)
    df = df.drop_duplicates(subset=["scientificName"], keep="last")

    return dict(zip(df["scientificName"], df["description"]))


def safe_bootstrap_assets():
    """
    Deja el sistema listo: modelo + encoder + mapa de descripciones.
    Lanza errores claros para cortar rápido si algo no cuadra.
    """
    # Modelo: descarga si falta
    download_release_asset_if_missing(XGB_MODEL_JSON_PATH, RELEASE_ASSET_NAME)
    if not XGB_MODEL_JSON_PATH.exists():
        raise FileNotFoundError(f"No existe el modelo en: {XGB_MODEL_JSON_PATH}")

    # Encoder
    if not ENCODER_PATH.exists():
        raise FileNotFoundError(f"No existe el label encoder en: {ENCODER_PATH}")

    # Descripciones
    desc_file = resolve_description_file(PROJECT_ROOT)
    desc_map = load_descriptions(desc_file)

    # Carga modelo
    model = XGBClassifier()
    model.load_model(str(XGB_MODEL_JSON_PATH))

    label_encoder = joblib.load(ENCODER_PATH)

    return model, label_encoder, desc_map, desc_file


# -----------------------------
# FUNCTIONS
# -----------------------------
def load_audio(file_path):
    waveform, _ = librosa.load(file_path, sr=TARGET_SR)
    return waveform


def compute_yamnet_embeddings(audio):
    scores, embeddings, spectrogram = YAMNET_MODEL(audio)
    emb_mean = tf.reduce_mean(embeddings, axis=0).numpy()
    emb_std = tf.math.reduce_std(embeddings, axis=0).numpy()
    n_frames = embeddings.shape[0]
    return np.concatenate([emb_mean, emb_std, [n_frames]])


def predict_top5(model, label_encoder, x):
    proba = model.predict_proba(np.array([x]))[0]
    top5_idx = np.argsort(proba)[::-1][:5]
    species = label_encoder.inverse_transform(top5_idx)
    scores = proba[top5_idx]
    return list(zip(species, scores))


# -----------------------------
# APP INIT (BLINDED)
# -----------------------------
BOOT_ERROR = None
model = None
label_encoder = None
desc_map = {}
desc_file_used = None

try:
    model, label_encoder, desc_map, desc_file_used = safe_bootstrap_assets()
except Exception as e:
    # No reventamos el servidor; mostramos el error en UI
    BOOT_ERROR = f"{type(e).__name__}: {e}"


# -----------------------------
# FLASK APP
# -----------------------------
app = Flask(
    __name__,
    template_folder="templates"   # la carpeta está dentro de src/templates
)

@app.route("/", methods=["GET", "POST"])
def index():
    # Si falló el arranque, lo mostramos directo
    if BOOT_ERROR:
        return render_template(
            "index.html",
            error="Fallo en el arranque de la app",
            details=BOOT_ERROR
        )

    if request.method == "POST":
        file = request.files.get("audio")
        if not file or file.filename == "":
            return render_template("index.html", error="No file selected")

        tmp_path = "temp_audio.wav"
        file.save(tmp_path)

        audio = load_audio(tmp_path)
        x = compute_yamnet_embeddings(audio)
        top5 = predict_top5(model, label_encoder, x)

        results = []
        for species, score in top5:
            desc = desc_map.get(species, "Description not available")
            results.append({
                "species": species,
                "score": round(float(score), 4),
                "description": desc
            })

        return render_template(
            "index.html",
            results=results,
            meta={
                "desc_file": str(desc_file_used),
                "classes": int(getattr(model, "n_classes_", 0)) or None
            }
        )

    return render_template(
        "index.html",
        meta={"desc_file": str(desc_file_used)} if desc_file_used else None
    )


if __name__ == "__main__":
    app.run(debug=True)




