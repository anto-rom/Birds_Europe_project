import os
import time
import uuid
import threading
import zipfile
from pathlib import Path

# Cache cross-platform (Render/Linux usa /tmp)
os.environ["TFHUB_CACHE_DIR"] = os.getenv(
    "TFHUB_CACHE_DIR",
    str(Path("/tmp") / "tfhub_cache")
)

from flask import Flask, request, render_template
from jinja2 import TemplateNotFound

import numpy as np
import pandas as pd
import joblib
import requests
from xgboost import XGBClassifier


# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# GitHub repo
GITHUB_OWNER = "anto-rom"
GITHUB_REPO = "Xeno_Canto_Project"

# Modelo XGB (ZIP en Releases -> descomprime a BST en /tmp)
RELEASE_ASSET_NAME = "xgb_model.bst.zip"
XGB_MODEL_ZIP_PATH = Path(os.getenv("XGB_MODEL_ZIP_PATH", str(Path("/tmp") / "xgb_model.bst.zip")))
XGB_MODEL_PATH = Path(os.getenv("XGB_MODEL_PATH", str(Path("/tmp") / "xgb_model.bst")))

# Encoder local (repo). Si no existe, fallback /tmp descargado desde Releases
ENCODER_PATH = BASE_DIR / "label_encoder.joblib"
ENCODER_TMP_PATH = Path("/tmp") / "label_encoder.joblib"
ENCODER_ASSET_NAME = "label_encoder.joblib"

# CSV de descripciones descargado desde DESC_CSV_URL (si está)
DESC_TMP_PATH = Path("/tmp") / "species_catalog_with_description.csv"

TARGET_SR = 16000


# -----------------------------
# YAMNet lazy-load (TF/TFHub lazy import)
# -----------------------------
YAMNET_MODEL = None

def get_yamnet():
    global YAMNET_MODEL
    if YAMNET_MODEL is None:
        import tensorflow_hub as hub
        # Evitar opciones experimentales en Render
        YAMNET_MODEL = hub.load("https://tfhub.dev/google/yamnet/1")
    return YAMNET_MODEL


# -----------------------------
# DOWNLOAD HELPERS
# -----------------------------
def download_release_asset_if_missing(local_path: Path, asset_name: str):
    """
    Descarga directa del asset del último release (sin GitHub API).
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
                f"GitHub devolvió HTML en vez de binario. "
                f"¿Existe '{asset_name}' en Releases?"
            )

        tmp = local_path.with_suffix(".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        tmp.replace(local_path)

    if not local_path.exists() or local_path.stat().st_size == 0:
        raise RuntimeError(f"Archivo descargado vacío/corrupto: {local_path}")


def download_url_if_missing(local_path: Path, url: str, timeout: int = 60):
    """
    Descarga desde una URL (raw github, etc.) a un fichero local.
    """
    if local_path.exists() and local_path.stat().st_size > 0:
        return

    local_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=timeout, allow_redirects=True) as r:
        r.raise_for_status()

        tmp = local_path.with_suffix(".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        tmp.replace(local_path)

    if not local_path.exists() or local_path.stat().st_size == 0:
        raise RuntimeError(f"Descarga vacía/corrupta desde {url}")


def unzip_if_missing(zip_path: Path, out_path: Path):
    """
    Descomprime zip_path en su directorio si out_path no existe.
    Espera que el zip contenga el fichero out_path.name.
    """
    if out_path.exists() and out_path.stat().st_size > 0:
        return

    zip_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        if out_path.name not in [Path(n).name for n in names]:
            raise RuntimeError(
                f"El zip no contiene {out_path.name}. Contiene (primeros): {names[:20]}"
            )
        z.extractall(zip_path.parent)

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"No se generó el modelo tras unzip: {out_path}")


# -----------------------------
# LOADERS
# -----------------------------
import requests

def ensure_xgb_model():
    if not XGB_MODEL_JSON_PATH.exists():
        print("⬇️ Descargando modelo XGBoost...")
        url = "https://raw.githubusercontent.com/anto-rom/Xeno_Canto_Project/main/src/xgb_model.json"   # AJUSTA ESTA URL
        r = requests.get(url)
        r.raise_for_status()

        with open(XGB_MODEL_JSON_PATH, "wb") as f:
            f.write(r.content)

        print("✅ Modelo descargado")

    else:
        print("✔️ Modelo ya existe en /tmp")


ensure_xgb_model()

# Ahora sí puedes cargarlo
model = XGBClassifier()
model.load_model(XGB_MODEL_JSON_PATH)


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

    expected = "\n- " + "\n- ".join(str(x) for x in candidates)
    raise FileNotFoundError(
        "No se encontró archivo de descripciones. Esperaba uno de:"
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
    if not required.issubset(set(df.columns)):
        raise ValueError(
            f"El archivo de descripciones debe tener columnas {required}. "
            f"Encontradas: {set(df.columns)}"
        )

    df["scientificName"] = df["scientificName"].astype(str).str.strip()
    df["description"] = df["description"].astype(str)

    return dict(zip(df["scientificName"], df["description"]))


# -----------------------------
# AUDIO / ML FUNCTIONS
# -----------------------------
def load_audio(file_path: Path):
    import librosa
    waveform, _ = librosa.load(str(file_path), sr=TARGET_SR)
    return waveform


def compute_yamnet_embeddings(audio):
    import tensorflow as tf
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
# APP + STATE
# -----------------------------
app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)

STATE = {
    "ready": False,
    "booting": False,
    "boot_error": None,

    "boot_step": None,
    "boot_started_at": None,

    "model": None,
    "label_encoder": None,
    "desc_map": None,
    "desc_file": None,
}

BOOT_LOCK = threading.Lock()


def _bootstrap():
    try:
        STATE["boot_started_at"] = time.time()
        STATE["boot_error"] = None

        # 1) Descargar ZIP del modelo desde Releases
        STATE["boot_step"] = "download_xgb_zip"
        download_release_asset_if_missing(XGB_MODEL_ZIP_PATH, RELEASE_ASSET_NAME)

        # 2) Unzip -> BST
        STATE["boot_step"] = "unzip_xgb"
        unzip_if_missing(XGB_MODEL_ZIP_PATH, XGB_MODEL_PATH)

        # 3) Cargar modelo
        STATE["boot_step"] = "load_xgb"
        model = XGBClassifier()
        model.load_model(str(XGB_MODEL_PATH))

        # 4) Encoder: repo si existe; si no, Releases
        STATE["boot_step"] = "ensure_encoder"
        encoder_path = ENCODER_PATH
        if (not encoder_path.exists()) or encoder_path.stat().st_size == 0:
            download_release_asset_if_missing(ENCODER_TMP_PATH, ENCODER_ASSET_NAME)
            encoder_path = ENCODER_TMP_PATH

        if (not encoder_path.exists()) or encoder_path.stat().st_size == 0:
            raise FileNotFoundError(f"Encoder no encontrado ni descargable: {encoder_path}")

        STATE["boot_step"] = "load_encoder"
        label_encoder = joblib.load(encoder_path)

        # 5) Descripciones: DESC_CSV_URL manda
        STATE["boot_step"] = "download_desc"
        desc_url = (os.getenv("DESC_CSV_URL") or "").strip()
        if desc_url:
            download_url_if_missing(DESC_TMP_PATH, desc_url, timeout=60)
            desc_file = DESC_TMP_PATH
        else:
            desc_file = resolve_description_file(PROJECT_ROOT)

        STATE["boot_step"] = "load_desc"
        desc_map = load_descriptions(desc_file)

        # Importante: NO cargar YAMNet aquí (dejarlo lazy)
        STATE["boot_step"] = "finalize"
        STATE["model"] = model
        STATE["label_encoder"] = label_encoder
        STATE["desc_map"] = desc_map
        STATE["desc_file"] = desc_file

        STATE["ready"] = True
        STATE["boot_step"] = "ready"

    except Exception as e:
        STATE["ready"] = False
        STATE["boot_error"] = f"{type(e).__name__}: {e}"
        STATE["boot_step"] = "error"
    finally:
        STATE["booting"] = False


def ensure_ready():
    if STATE["ready"] or STATE["booting"]:
        return

    with BOOT_LOCK:
        if STATE["ready"] or STATE["booting"]:
            return
        STATE["booting"] = True
        t = threading.Thread(target=_bootstrap, daemon=True)
        t.start()


# -----------------------------
# ROUTES (limpias y definitivas)
# -----------------------------
@app.route("/healthz", methods=["GET"])
def healthz():
    elapsed = None
    if STATE.get("boot_started_at"):
        elapsed = round(time.time() - STATE["boot_started_at"], 1)

    return {
        "ready": STATE["ready"],
        "booting": STATE["booting"],
        "boot_step": STATE.get("boot_step"),
        "boot_error": STATE["boot_error"],
        "boot_elapsed_s": elapsed,
    }, 200


@app.route("/debug_fs", methods=["GET"])
def debug_fs():
    templates_dir = BASE_DIR / "templates"
    static_dir = BASE_DIR / "static"

    def safe_list(p: Path):
        try:
            return sorted(os.listdir(p))
        except Exception as e:
            return [f"ERROR: {type(e).__name__}: {e}"]

    return {
        "cwd": str(Path.cwd()),
        "base_dir": str(BASE_DIR),
        "templates_dir": str(templates_dir),
        "templates_exists": templates_dir.exists(),
        "templates_list": safe_list(templates_dir),
        "static_dir": str(static_dir),
        "static_exists": static_dir.exists(),
        "static_list": safe_list(static_dir),
        "xgb_zip_path": str(XGB_MODEL_ZIP_PATH),
        "xgb_zip_exists": XGB_MODEL_ZIP_PATH.exists(),
        "xgb_path": str(XGB_MODEL_PATH),
        "xgb_exists": XGB_MODEL_PATH.exists(),
    }, 200


@app.route("/warmup", methods=["GET"])
def warmup():
    ensure_ready()

    elapsed = None
    if STATE.get("boot_started_at"):
        elapsed = round(time.time() - STATE["boot_started_at"], 1)

    return {
        "ready": STATE["ready"],
        "booting": STATE["booting"],
        "boot_step": STATE.get("boot_step"),
        "boot_error": STATE["boot_error"],
        "boot_elapsed_s": elapsed,
    }, 200


@app.route("/", methods=["GET", "POST", "HEAD"])
def index():
    # Render health check hace HEAD / (Go-http-client). No dispares bootstrap aquí.
    if request.method == "HEAD":
        return ("", 200)

    ensure_ready()

    if not STATE["ready"]:
        try:
            return render_template(
                "index.html",
                error="Inicializando..." if STATE["booting"] else "Fallo en el arranque de la app",
                details=STATE["boot_error"],
                meta={
                    "boot_step": STATE.get("boot_step"),
                    "booting": STATE["booting"],
                    "ready": STATE["ready"],
                    "desc_file": str(STATE["desc_file"]) if STATE["desc_file"] else None,
                },
            )
        except TemplateNotFound:
            expected = BASE_DIR / "templates" / "index.html"
            return (f"TemplateNotFound: index.html (expected at {expected})", 500)

    # Listo → lógica principal
    model = STATE["model"]
    label_encoder = STATE["label_encoder"]
    desc_map = STATE["desc_map"]
    desc_file_used = STATE["desc_file"]

    if request.method == "POST":
        f = request.files.get("audio")
        if not f:
            return render_template(
                "index.html",
                error="No se subió ningún archivo",
                meta={"desc_file": str(desc_file_used)},
            )

        tmp_path = Path("/tmp") / f"{uuid.uuid4().hex}_{f.filename}"
        f.save(tmp_path)

        try:
            waveform = load_audio(tmp_path)
            x = compute_yamnet_embeddings(waveform)
            top5 = predict_top5(model, label_encoder, x)

            results = []
            for sp, score in top5:
                results.append({
                    "scientificName": sp,
                    "score": float(score),
                    "description": desc_map.get(sp, "No description available."),
                })

        except Exception as e:
            return render_template(
                "index.html",
                error="Error procesando el audio",
                details=f"{type(e).__name__}: {e}",
                meta={"desc_file": str(desc_file_used)},
            )
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

        return render_template(
            "index.html",
            results=results,
            meta={"desc_file": str(desc_file_used)},
        )

    return render_template(
        "index.html",
        meta={"desc_file": str(desc_file_used)},
    )


if __name__ == "__main__":
    # Local only
    app.run(debug=True)

