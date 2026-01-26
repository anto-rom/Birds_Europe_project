import os
import time
import uuid
import threading
import traceback
from pathlib import Path
import csv
import json

import numpy as np
import requests
from flask import Flask, request, render_template
from jinja2 import TemplateNotFound
from werkzeug.utils import secure_filename


# ------------------------------------------
# LOGGING
# ------------------------------------------
def log(msg: str):
    print(f"[APP] {msg}", flush=True)


# Cache cross-platform (Render/Linux usa /tmp)
os.environ["TFHUB_CACHE_DIR"] = os.getenv("TFHUB_CACHE_DIR", str(Path("/tmp") / "tfhub_cache"))


# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

GITHUB_OWNER = os.getenv("GITHUB_OWNER", "anto-rom")
GITHUB_REPO = os.getenv("GITHUB_REPO", "Xeno_Canto_Project")

# Recomendado: tag fijo para evitar sorpresas con "latest"
RELEASE_TAG = os.getenv("RELEASE_TAG", "").strip()  # ej: "v1.1-compact"
RELEASE_ASSET_NAME = os.getenv("RELEASE_ASSET_NAME", "compact_xgb_model.ubj")

# Descarga a /tmp (modelo directo; NO zip por memoria)
XGB_MODEL_LOCAL = Path(os.getenv("XGB_MODEL_PATH", str(Path("/tmp") / RELEASE_ASSET_NAME)))

# Clases (ligero, sin sklearn/joblib)
CLASSES_ASSET_NAME = os.getenv("CLASSES_ASSET_NAME", "classes.json")
CLASSES_PATH_REPO = BASE_DIR / "classes.json"
CLASSES_PATH_TMP = Path("/tmp") / "classes.json"

TARGET_SR = 16000

# Upload hardening
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))  # ajusta si quieres
MAX_CONTENT_LENGTH = MAX_UPLOAD_MB * 1024 * 1024


# -----------------------------
# YAMNet lazy-load
# -----------------------------
YAMNET_MODEL = None


def get_yamnet():
    global YAMNET_MODEL
    if YAMNET_MODEL is None:
        import tensorflow_hub as hub  # lazy
        log("Lazy-loading YAMNet (tfhub)...")
        YAMNET_MODEL = hub.load("https://tfhub.dev/google/yamnet/1")
        log("YAMNet loaded.")
    return YAMNET_MODEL


# -----------------------------
# DOWNLOAD HELPERS
# -----------------------------
def _default_headers():
    return {"User-Agent": "render-bird-app/1.0", "Accept": "*/*"}


def _assert_not_html_response(r: requests.Response, url: str, asset_hint: str = ""):
    ctype = (r.headers.get("Content-Type") or "").lower()
    if "text/html" in ctype:
        raise RuntimeError(
            "GitHub/URL devolvió HTML en vez de binario. "
            f"{asset_hint} URL={url} content-type={ctype}"
        )


def _release_download_url(asset_name: str) -> str:
    """
    Si RELEASE_TAG está definido -> descarga determinista desde ese tag.
    Si no -> cae a latest (menos determinista).
    """
    if RELEASE_TAG:
        return f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/releases/download/{RELEASE_TAG}/{asset_name}"
    return f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest/download/{asset_name}"


def download_release_asset_if_missing(local_path: Path, asset_name: str):
    if local_path.exists() and local_path.stat().st_size > 0:
        log(f"Asset ya existe: {local_path} ({local_path.stat().st_size} bytes)")
        return

    url = _release_download_url(asset_name)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    log(f"Downloading release asset: {asset_name}")
    log(f"URL:  {url}")
    log(f"Dest: {local_path}")

    with requests.get(
        url,
        stream=True,
        timeout=(10, 180),
        allow_redirects=True,
        headers=_default_headers()
    ) as r:
        r.raise_for_status()
        _assert_not_html_response(r, url, asset_hint=f"asset='{asset_name}'")

        tmp = local_path.with_suffix(".part")
        bytes_written = 0

        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bytes_written += len(chunk)

        tmp.replace(local_path)

    if not local_path.exists() or local_path.stat().st_size == 0:
        raise RuntimeError(f"Archivo descargado vacío/corrupto: {local_path}")

    log(f"Downloaded OK: {local_path} ({local_path.stat().st_size} bytes, wrote={bytes_written})")


def prepare_model_file(asset_name: str) -> Path:
    """
    SOLO soporta modelo directo (.ubj, .json, .bst).
    No soporta .zip — evita unzip y picos de memoria.
    """
    if asset_name.lower().endswith(".zip"):
        raise RuntimeError(
            f"Asset .zip ({asset_name}) no soportado por memoria. "
            "Sube un archivo directo (.ubj/.json/.bst)."
        )

    local_asset = Path("/tmp") / asset_name
    download_release_asset_if_missing(local_asset, asset_name)

    if not local_asset.exists() or local_asset.stat().st_size == 0:
        raise RuntimeError(f"Modelo no encontrado o vacío: {local_asset}")

    return local_asset


# -----------------------------
# DESCRIPTIONS
# -----------------------------
def resolve_description_file(project_root: Path) -> Path:
    """
    Busca el CSV de descripciones en rutas típicas del repo.
    """
    candidates = [
        project_root / "data" / "processed" / "species_catalog_with_description.csv",
        project_root / "src" / "species_catalog_with_description.csv",
        BASE_DIR / "species_catalog_with_description.csv",
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 0:
            return p

    expected = "\n- " + "\n- ".join(str(x) for x in candidates)
    raise FileNotFoundError("No se encontró archivo de descripciones. Esperaba uno de:" f"{expected}")


def load_descriptions(file_path: Path) -> dict:
    """
    Carga un CSV con columnas: scientificName, description
    Acepta delimitador ',' o ';' y encodings típicos.
    """
    if file_path.suffix.lower() != ".csv":
        raise ValueError(f"Solo se admite CSV para descripciones: {file_path}")

    for delimiter in [",", ";"]:
        for enc in ["utf-8", "latin1"]:
            try:
                with open(file_path, "r", encoding=enc, newline="") as f:
                    reader = csv.DictReader(f, delimiter=delimiter)
                    if not reader.fieldnames:
                        continue

                    cols = {c.strip() for c in reader.fieldnames}
                    if not {"scientificName", "description"}.issubset(cols):
                        continue

                    out = {}
                    for row in reader:
                        sp = (row.get("scientificName") or "").strip()
                        desc = (row.get("description") or "").strip()
                        if sp:
                            out[sp] = desc
                    if out:
                        return out
            except UnicodeDecodeError:
                continue

    raise ValueError(
        f"CSV de descripciones inválido: {file_path}. "
        "Debe tener columnas scientificName y description (separador ',' o ';')."
    )


# -----------------------------
# AUDIO / ML
# -----------------------------
def allowed_file(filename: str) -> bool:
    suffix = Path(filename).suffix.lower()
    return suffix in ALLOWED_EXTENSIONS


def load_audio(file_path: Path):
    """
    Carga audio resampleado a TARGET_SR.
    """
    import librosa  # lazy
    waveform, _ = librosa.load(str(file_path), sr=TARGET_SR, mono=True)
    return waveform


    x = compute_yamnet_embeddings(waveform)
    top5 = predict_top5(booster, classes, x)

def compute_yamnet_embeddings(audio: np.ndarray) -> np.ndarray:
    """
    Embedding agregado: mean + std + n_frames
    """
    import tensorflow as tf  # lazy
    yamnet = get_yamnet()

    audio_tf = tf.convert_to_tensor(audio, dtype=tf.float32)
    audio_tf = tf.reshape(audio_tf, [-1])

    _, embeddings, _ = yamnet(audio_tf)
    emb_mean = tf.reduce_mean(embeddings, axis=0).numpy()
    emb_std = tf.math.reduce_std(embeddings, axis=0).numpy()
    n_frames = int(embeddings.shape[0])

    return np.concatenate([emb_mean, emb_std, [n_frames]]).astype(np.float32)


def predict_top5(booster, classes: list[str], x: np.ndarray):
    import xgboost as xgb  # lazy

    X = np.asarray([x], dtype=np.float32)
    dm = xgb.DMatrix(X)
    pred = booster.predict(dm)

    pred = np.asarray(pred)
    if pred.ndim != 2:
        raise RuntimeError(
            f"Formato de predicción inesperado: shape={pred.shape}. "
            "Entrena con objective='multi:softprob'."
        )

    proba = pred[0]
    idx = np.argsort(proba)[::-1][:5]

    species = [classes[i] if i < len(classes) else f"class_{i}" for i in idx]
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
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

STATE = {
    "ready": False,
    "booting": False,
    "boot_error": None,
    "boot_step": None,
    "boot_started_at": None,
    "booster": None,
    "classes": None,
    "desc_map": None,
    "desc_file": None,
}

BOOT_LOCK = threading.Lock()
BOOT_THREAD = None


def _set_state(step=None, err=None, ready=None, booting=None):
    if step is not None:
        STATE["boot_step"] = step
    if err is not None:
        STATE["boot_error"] = err
    if ready is not None:
        STATE["ready"] = ready
    if booting is not None:
        STATE["booting"] = booting


def _bootstrap():
    try:
        _set_state(err=None)
        log(f"BOOT start. RELEASE_TAG={RELEASE_TAG or 'latest'} ASSET={RELEASE_ASSET_NAME}")

        # 1) Modelo
        _set_state(step="download_model")
        model_path = prepare_model_file(RELEASE_ASSET_NAME)

        # 2) Booster
        _set_state(step="load_xgb")
        import xgboost as xgb  # lazy
        log(f"Loading Booster from {model_path} size={model_path.stat().st_size}")
        booster = xgb.Booster()
        booster.load_model(str(model_path))
        log("Booster loaded OK.")

        # 3) Clases
        _set_state(step="ensure_classes")
        classes_path = CLASSES_PATH_REPO
        if (not classes_path.exists()) or classes_path.stat().st_size == 0:
            log(f"classes.json not found in repo: {CLASSES_PATH_REPO}. Downloading from release...")
            download_release_asset_if_missing(CLASSES_PATH_TMP, CLASSES_ASSET_NAME)
            classes_path = CLASSES_PATH_TMP

        _set_state(step="load_classes")
        with open(classes_path, "r", encoding="utf-8") as f:
            classes = json.load(f)
        if not isinstance(classes, list) or not classes:
            raise ValueError(f"classes.json inválido: {classes_path}")
        log(f"Classes loaded: {classes_path} (n={len(classes)})")

        # 4) Descripciones (local repo)
        _set_state(step="load_desc")
        desc_file = resolve_description_file(PROJECT_ROOT)
        desc_map = load_descriptions(desc_file)
        log(f"Descriptions loaded from: {desc_file} (n={len(desc_map)})")

        STATE["booster"] = booster
        STATE["classes"] = classes
        STATE["desc_map"] = desc_map
        STATE["desc_file"] = desc_file

        _set_state(ready=True, step="ready")
        log("BOOT ready.")

    except Exception as e:
        tb = traceback.format_exc()
        _set_state(ready=False, step="error", err=f"{type(e).__name__}: {e}\n{tb}")
        log(f"BOOT error: {type(e).__name__}: {e}")
    finally:
        _set_state(booting=False)


def ensure_ready():
    """
    Dispara el bootstrap una única vez (thread-safe).
    """
    global BOOT_THREAD
    if STATE["ready"] or STATE["booting"]:
        return

    with BOOT_LOCK:
        if STATE["ready"] or STATE["booting"]:
            return

        STATE["boot_started_at"] = time.time()
        _set_state(booting=True, ready=False, err=None, step="start")
        BOOT_THREAD = threading.Thread(target=_bootstrap, daemon=True)
        BOOT_THREAD.start()


# -----------------------------
# ROUTES
# -----------------------------
@app.route("/healthz", methods=["GET"])
def healthz():
    # Ojo: si no quieres que el healthcheck dispare carga pesada, comenta ensure_ready()
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
        "release_tag": RELEASE_TAG or "latest",
        "asset": RELEASE_ASSET_NAME,
        "max_upload_mb": MAX_UPLOAD_MB,
    }, 200


@app.route("/warmup", methods=["GET"])
def warmup():
    ensure_ready()
    return {
        "ok": True,
        "ready": STATE["ready"],
        "boot_step": STATE.get("boot_step"),
        "boot_error": STATE["boot_error"],
    }, 200


@app.route("/", methods=["GET", "POST", "HEAD"])
def index():
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
                },
            )
        except TemplateNotFound:
            expected = BASE_DIR / "templates" / "index.html"
            return (f"TemplateNotFound: index.html (expected at {expected})", 500)

    booster = STATE["booster"]
    classes = STATE["classes"]
    desc_map = STATE["desc_map"]
    desc_file_used = STATE["desc_file"]

    if request.method == "POST":
        f = request.files.get("audio")
        if not f or not f.filename:
            return render_template(
                "index.html",
                error="No se subió ningún archivo",
                meta={"desc_file": str(desc_file_used)},
            )

        filename = secure_filename(f.filename)
        if not allowed_file(filename):
            return render_template(
                "index.html",
                error=f"Formato no soportado. Sube: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
                meta={"desc_file": str(desc_file_used)},
            )

        tmp_path = Path("/tmp") / f"{uuid.uuid4().hex}_{filename}"
        f.save(tmp_path)

        try:
            waveform = load_audio(tmp_path)

            MAX_SECONDS = int(os.getenv("MAX_AUDIO_SECONDS", "10"))
            max_len = TARGET_SR * MAX_SECONDS
            if len(waveform) > max_len:
               waveform = waveform[:max_len]

            # LOG: ver cuántos segundos reales se van a procesar
            log(f"Audio samples={len(waveform)} seconds={len(waveform)/TARGET_SR:.2f}")

            x = compute_yamnet_embeddings(waveform)
            top5 = predict_top5(booster, classes, x)

            results = [
                {
                    "scientificName": sp,
                    "score": float(score),
                    "description": desc_map.get(sp, "No description available."),
                }
                for sp, score in top5
            ]

 

        except Exception as e:
            return render_template(
                "index.html",
                error="Error procesando el audio",
                details=f"{type(e).__name__}: {e}",
            )
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

        return render_template("index.html", results=results, meta={"desc_file": str(desc_file_used)})

    return render_template("index.html", meta={"desc_file": str(desc_file_used)})


if __name__ == "__main__":
    # Render usa $PORT
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)







