import os
from pathlib import Path
import uuid
import threading

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

# En Render descargamos el modelo a /tmp
XGB_MODEL_JSON_PATH = Path(
    os.getenv("XGB_MODEL_PATH", str(Path("/tmp") / "xgb_model.json"))
)

ENCODER_PATH = BASE_DIR / "label_encoder.joblib"

GITHUB_OWNER = "anto-rom"
GITHUB_REPO = "Xeno_Canto_Project"
RELEASE_ASSET_NAME = "xgb_model.json"

TARGET_SR = 16000

# -----------------------------
# YAMNet lazy-load
# -----------------------------
YAMNET_MODEL = None

def get_yamnet():
    global YAMNET_MODEL
    if YAMNET_MODEL is None:
        load_opts = tf.saved_model.LoadOptions(experimental_io_device="/job:localhost")
        YAMNET_MODEL = hub.load("https://tfhub.dev/google/yamnet/1", options=load_opts)
    return YAMNET_MODEL

# -----------------------------
# DOWNLOAD HELPERS (sin API)
# -----------------------------
def download_release_asset_if_missing(local_path: Path, asset_name: str):
    """
    Descarga directa del asset del último release (sin GitHub API).
    Evita rate limit y simplifica.
    """
    if local_path.exists() and local_path.stat().st_size > 0:
        return

    url = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest/download/{asset_name}"
    local_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=180, allow_redirects=True) as r:
        r.raise_for_status()

        ctype = (r.headers.get("Content-Type") or "").lower()
        if "text/html" in ctype:
            raise RuntimeError(
                f"GitHub devolvió HTML, no binario. "
                f"¿Existe '{asset_name}' en Releases?"
            )

        tmp = local_path.with_suffix(".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        tmp.replace(local_path)

    if not local_path.exists() or local_path.stat().st_size == 0:
        raise RuntimeError("Archivo descargado vacío/corrupto.")

# -----------------------------
# LOADERS
# -----------------------------
def resolve_description_file(project_root: Path) -> Path:
    candidates = [
        project_root / "data" / "raw" / "scientificName_description.xlsx",
        project_root / "data" / "raw" / "species_catalog_with_description.xlsx",
        project_root / "data" / "processed" / "species_catalog_with_description.xlsx",
        project_root / "data" / "processed" / "species_catalog_with_description.csv",
        project_root / "src" / "species_catalog_with_description.csv",
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 0:
            return p

    expected = "\n".join([str(p) for p in candidates])
    raise FileNotFoundError(
        "No encuentro archivo de descripciones. Busqué en:\n"
        f"{expected}"
    )

def load_descriptions(file_path: Path) -> dict:
    if file_path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    else:
        try:
            df = pd.read_csv(file_path)
        except Exception:
            df = pd.read_csv(file_path, sep=";")

    df.columns = [c.strip() for c in df.columns]

    required = {"scientificName", "description"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Columnas requeridas no encontradas en {file_path.name}. "
            f"Tengo: {df.columns}"
        )

    df = df[["scientificName", "description"]].copy()
    df["scientificName"] = df["scientificName"].astype(str).str.strip()
    df["description"] = df["description"].astype(str).str.strip()

    df = df[df["scientificName"] != ""]
    df = df[df["description"] != ""]
    df = df.drop_duplicates(subset=["scientificName"], keep="last")

    return dict(zip(df["scientificName"], df["description"]))

def safe_bootstrap_assets():
    download_release_asset_if_missing(XGB_MODEL_JSON_PATH, RELEASE_ASSET_NAME)
    if not XGB_MODEL_JSON_PATH.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {XGB_MODEL_JSON_PATH}")

    if not ENCODER_PATH.exists():
        raise FileNotFoundError(
            f"Encoder no encontrado: {ENCODER_PATH}. "
            "Acción: súbelo a src/ o añade descarga."
        )

    desc_file = resolve_description_file(PROJECT_ROOT)
    desc_map = load_descriptions(desc_file)

    model = XGBClassifier()
    model.load_model(str(XGB_MODEL_JSON_PATH))

    label_encoder = joblib.load(ENCODER_PATH)

    return model, label_encoder, desc_map, desc_file

# -----------------------------
# AUDIO / ML FUNCTIONS
# -----------------------------
def load_audio(file_path):
    waveform, _ = librosa.load(str(file_path), sr=TARGET_SR)
    return waveform

def compute_yamnet_embeddings(audio):
    yamnet = get_yamnet()
    scores, embeddings, spectrogram = yamnet(audio)
    emb_mean = tf.reduce_mean(embeddings, axis=0).numpy()
    emb_std = tf.math.reduce_std(embeddings, axis=0).numpy()
    n_frames = embeddings.shape[0]
    return np.concatenate([emb_mean, emb_std, [n_frames]])

def predict_top5(model, label_encoder, x):
    proba = model.predict_proba([x])[0]
    idx = np.argsort(proba)[::-1][:5]
    species = label_encoder.inverse_transform(idx)
    scores = proba[idx]
    return list(zip(species, scores))

# -----------------------------
# LAZY BOOTSTRAP (PRO-RENDER)
# -----------------------------
STATE = {
    "ready": False,
    "boot_error": None,
    "model": None,
    "label_encoder": None,
    "desc_map": {},
    "desc_file_used": None,
}
_BOOT_LOCK = threading.Lock()

def ensure_ready():
    if STATE["ready"]:
        return

    with _BOOT_LOCK:
        if STATE["ready"]:
            return

        try:
            m, le, dm, df = safe_bootstrap_assets()
            STATE["model"] = m
            STATE["label_encoder"] = le
            STATE["desc_map"] = dm
            STATE["desc_file_used"] = df
            STATE["ready"] = True
            STATE["boot_error"] = None
        except Exception as e:
            STATE["ready"] = False
            STATE["boot_error"] = f"{type(e).__name__}: {e} | repo={GITHUB_OWNER}/{GITHUB_REPO}"


# -----------------------------
# FLASK APP
# -----------------------------
app = Flask(__name__, template_folder="templates")

@app.get("/debug_boot")
def debug_boot():
    # No expone secretos, solo configuración básica
    return {
        "github_owner": GITHUB_OWNER,
        "github_repo": GITHUB_REPO,
        "asset": RELEASE_ASSET_NAME,
        "download_url_example": f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest/download/{RELEASE_ASSET_NAME}",
        "uses_api": False,
    }, 200


@app.get("/healthz")
def healthz():
    ensure_ready()
    if STATE["ready"]:
        return {"status": "ready"}, 200
    return {"status": "error", "detail": STATE["boot_error"]}, 500

@app.route("/", methods=["GET", "POST"])
def index():
    ensure_ready()

    if not STATE["ready"]:
        return render_template(
            "index.html",
            error="Fallo en el arranque de la app",
            details=STATE["boot_error"]
        )

    model = STATE["model"]
    label_encoder = STATE["label_encoder"]
    desc_map = STATE["desc_map"]
    desc_file_used = STATE["desc_file_used"]

    if request.method == "POST":
        file = request.files.get("audio")
        if not file or file.filename == "":
            return render_template("index.html", error="No file selected")

        tmp_path = Path("/tmp") / f"upload_{uuid.uuid4().hex}.wav"
        file.save(str(tmp_path))

        try:
            audio = load_audio(tmp_path)
            x = compute_yamnet_embeddings(audio)
            top5 = predict_top5(model, label_encoder, x)
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except:
                pass

        results = []
        for species, score in top5:
            desc = desc_map.get(species, "Description not available")
            results.append({
                "species": species,
                "score": float(score),
                "description": desc
            })

        return render_template(
            "index.html",
            results=results,
            meta={"desc_file": str(desc_file_used)}
        )

    return render_template(
        "index.html",
        meta={"desc_file": str(desc_file_used)}
    )

if __name__ == "__main__":
    app.run(debug=True)

