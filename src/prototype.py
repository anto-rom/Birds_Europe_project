import os
from pathlib import Path
import uuid
import threading
from jinja2 import TemplateNotFound

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

# Encoder local (repo). Si no existe, usamos fallback /tmp descargado desde Releases
ENCODER_PATH = BASE_DIR / "label_encoder.joblib"
ENCODER_TMP_PATH = Path("/tmp") / "label_encoder.joblib"

# CSV de descripciones descargado desde DESC_CSV_URL (si está)
DESC_TMP_PATH = Path("/tmp") / "species_catalog_with_description.csv"

GITHUB_OWNER = "anto-rom"
GITHUB_REPO = "Xeno_Canto_Project"

RELEASE_ASSET_NAME = "xgb_model.json"
ENCODER_ASSET_NAME = "label_encoder.joblib"

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


def download_url_if_missing(local_path: Path, url: str, timeout: int = 180):
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


def safe_bootstrap_assets():
    """
    - Descarga modelo XGB desde Releases a /tmp si falta
    - Descarga encoder desde Releases a /tmp si falta en repo
    - Descarga CSV de descripciones desde DESC_CSV_URL si existe; si no, usa archivo local
    """
    # 1) Modelo XGBoost desde Releases
    download_release_asset_if_missing(XGB_MODEL_JSON_PATH, RELEASE_ASSET_NAME)
    if not XGB_MODEL_JSON_PATH.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {XGB_MODEL_JSON_PATH}")

    # 2) Encoder: usa repo si existe; si no, baja de Releases
    encoder_path = ENCODER_PATH
    if (not encoder_path.exists()) or encoder_path.stat().st_size == 0:
        download_release_asset_if_missing(ENCODER_TMP_PATH, ENCODER_ASSET_NAME)
        encoder_path = ENCODER_TMP_PATH

    if (not encoder_path.exists()) or encoder_path.stat().st_size == 0:
        raise FileNotFoundError(f"Encoder no encontrado ni descargable: {encoder_path}")

    # 3) Descripciones
    desc_url = (os.getenv("DESC_CSV_URL") or "").strip()
    if desc_url:
        download_url_if_missing(DESC_TMP_PATH, desc_url, timeout=180)
        desc_file = DESC_TMP_PATH
    else:
        desc_file = resolve_description_file(PROJECT_ROOT)

    desc_map = load_descriptions(desc_file)

    # 4) Cargar modelo y encoder
    model = XGBClassifier()
    model.load_model(str(XGB_MODEL_JSON_PATH))

    label_encoder = joblib.load(encoder_path)

    return model, label_encoder, desc_map, desc_file


# -----------------------------
# AUDIO / ML FUNCTIONS
# -----------------------------
def load_audio(file_path: Path):
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
# APP + STATE
# -----------------------------
app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static")
)


STATE = {
    "ready": False,
    "booting": False,
    "boot_error": None,
    "model": None,
    "label_encoder": None,
    "desc_map": None,
    "desc_file": None,
}

BOOT_LOCK = threading.Lock()


def _bootstrap():
    try:
        print("BOOT: starting bootstrap...")
        m, le, dm, df = safe_bootstrap_assets()
        print("BOOT: assets loaded OK")
        STATE["model"] = m
        STATE["label_encoder"] = le
        STATE["desc_map"] = dm
        STATE["desc_file"] = df
        STATE["ready"] = True
        STATE["boot_error"] = None
        print("BOOT: READY")
    except Exception as e:
        STATE["ready"] = False
        STATE["boot_error"] = f"{type(e).__name__}: {e}"
        print("BOOT: ERROR ->", STATE["boot_error"])
    finally:
        STATE["booting"] = False


def ensure_ready():
    # No bloquear request; dispara bootstrap en background una vez
    if STATE["ready"]:
        return
    if STATE["booting"]:
        return

    with BOOT_LOCK:
        if STATE["ready"] or STATE["booting"]:
            return
        STATE["booting"] = True
        t = threading.Thread(target=_bootstrap, daemon=True)
        t.start()


# -----------------------------
# ROUTES
# -----------------------------
@app.route("/healthz", methods=["GET"])
def healthz():
    ensure_ready()

    payload = {
        "status": "ready" if STATE["ready"] else "starting" if STATE["booting"] else "error",
        "booting": STATE["booting"],
        "ready": STATE["ready"],
        "boot_error": STATE["boot_error"],
        "desc_file": str(STATE["desc_file"]) if STATE.get("desc_file") else None,
        "xgb_model_path": str(XGB_MODEL_JSON_PATH),
        "xgb_model_exists": XGB_MODEL_JSON_PATH.exists(),
        "xgb_model_size_mb": round(XGB_MODEL_JSON_PATH.stat().st_size / 1024 / 1024, 2) if XGB_MODEL_JSON_PATH.exists() else None,
        "encoder_repo_exists": ENCODER_PATH.exists(),
        "encoder_tmp_exists": (Path("/tmp") / "label_encoder.joblib").exists(),
        "desc_csv_url": os.getenv("DESC_CSV_URL"),
    }

    code = 200 if STATE["ready"] else 503
    return payload, code


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
    }, 200


@app.route("/debug_boot", methods=["GET"])
def debug_boot():
    return {
        "cwd": str(Path.cwd()),
        "base_dir": str(BASE_DIR),
        "project_root": str(PROJECT_ROOT),
        "xgb_model_path": str(XGB_MODEL_JSON_PATH),
        "encoder_path_repo": str(ENCODER_PATH),
        "encoder_path_tmp": str(ENCODER_TMP_PATH),
        "desc_csv_url": os.getenv("DESC_CSV_URL"),
        "tfhub_cache_dir": os.getenv("TFHUB_CACHE_DIR"),
        "ready": STATE["ready"],
        "booting": STATE["booting"],
        "boot_error": STATE["boot_error"],
    }, 200

@app.route("/", methods=["GET", "POST"])
def index():
    ensure_ready()

    try:
        if not STATE["ready"]:
            return render_template(
                "index.html",
                error="Fallo en el arranque de la app",
                details=STATE["boot_error"],
                meta={"desc_file": str(STATE["desc_file"]) if STATE["desc_file"] else None}
            )

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
                    meta={"desc_file": str(desc_file_used)}
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
                        "description": desc_map.get(sp, "No description available.")
                    })

            except Exception as e:
                return render_template(
                    "index.html",
                    error="Error procesando el audio",
                    details=f"{type(e).__name__}: {e}",
                    meta={"desc_file": str(desc_file_used)}
                )
            finally:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass

            return render_template(
                "index.html",
                results=results,
                meta={"desc_file": str(desc_file_used)}
            )

        return render_template(
            "index.html",
            meta={"desc_file": str(desc_file_used)}
        )

    except TemplateNotFound as e:
        # Esto te da visibilidad brutal y corta el "Bad Gateway"
        return (
            f"TemplateNotFound: {e}. Expected: {BASE_DIR / 'templates' / 'index.html'}",
            500
        )

if __name__ == "__main__":
    app.run(debug=True)
