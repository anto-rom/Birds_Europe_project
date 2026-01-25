import os
import time
import uuid
import threading
import zipfile
import traceback
from pathlib import Path

import numpy as np
import csv
import json
import requests
from flask import Flask, request, render_template
from jinja2 import TemplateNotFound

import xgboost as xgb  # <-- Booster/DMatrix


# ------------------------------------------
# RENDER / LOGGING
# ------------------------------------------
# Recomendado en Render como env var: PYTHONUNBUFFERED=1
def log(msg: str):
    print(f"[BOOT] {msg}", flush=True)


# Cache cross-platform (Render/Linux usa /tmp)
os.environ["TFHUB_CACHE_DIR"] = os.getenv("TFHUB_CACHE_DIR", str(Path("/tmp") / "tfhub_cache"))

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# GitHub repo (mejor por env vars para no redeployar)
GITHUB_OWNER = os.getenv("GITHUB_OWNER", "anto-rom")
GITHUB_REPO = os.getenv("GITHUB_REPO", "Xeno_Canto_Project")

# Modelo XGB (ZIP en Releases -> descomprime a BST en /tmp)
RELEASE_ASSET_NAME = os.getenv("RELEASE_ASSET_NAME", "xgb_model.bst.zip")
XGB_MODEL_ZIP_PATH = Path(os.getenv("XGB_MODEL_ZIP_PATH", str(Path("/tmp") / RELEASE_ASSET_NAME)))
XGB_MODEL_PATH = Path(os.getenv("XGB_MODEL_PATH", str(Path("/tmp") / "xgb_model.bst")))

# Clases (evita cargar sklearn/joblib en Render: mucho consumo de RAM)
# Genera este archivo en entrenamiento con algo como:
#   json.dump(label_encoder.classes_.tolist(), open('classes.json','w'))
CLASSES_ASSET_NAME = os.getenv("CLASSES_ASSET_NAME", "classes.json")
CLASSES_PATH = BASE_DIR / "classes.json"
CLASSES_TMP_PATH = Path("/tmp") / "classes.json"

# CSV de descripciones descargado desde DESC_CSV_URL (si está)
DESC_TMP_PATH = Path("/tmp") / "species_catalog_with_description.csv"

TARGET_SR = 16000


# -----------------------------
# YAMNet lazy-load
# -----------------------------
YAMNET_MODEL = None

def get_yamnet():
    global YAMNET_MODEL
    if YAMNET_MODEL is None:
        import tensorflow_hub as hub
        log("Lazy-loading YAMNet (tfhub)...")
        YAMNET_MODEL = hub.load("https://tfhub.dev/google/yamnet/1")
        log("YAMNet loaded.")
    return YAMNET_MODEL


# -----------------------------
# DOWNLOAD HELPERS
# -----------------------------
def _default_headers():
    return {
        "User-Agent": "render-bird-app/1.0",
        "Accept": "*/*",
    }

def _assert_not_html_response(r: requests.Response, url: str, asset_hint: str = ""):
    ctype = (r.headers.get("Content-Type") or "").lower()
    if "text/html" in ctype:
        raise RuntimeError(
            "GitHub/URL devolvió HTML en vez de binario. "
            f"{asset_hint} URL={url} content-type={ctype}"
        )

def download_release_asset_if_missing(local_path: Path, asset_name: str):
    """
    Descarga directa del asset del último release (sin GitHub API).
    """
    if local_path.exists() and local_path.stat().st_size > 0:
        log(f"Asset ya existe: {local_path} ({local_path.stat().st_size} bytes)")
        return

    url = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest/download/{asset_name}"
    local_path.parent.mkdir(parents=True, exist_ok=True)

    log(f"Downloading release asset: {asset_name}")
    log(f"URL: {url}")
    log(f"Dest: {local_path}")

    with requests.get(
        url,
        stream=True,
        timeout=(10, 120),
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


def download_url_if_missing(local_path: Path, url: str, timeout: int = 60):
    """
    Descarga desde una URL a un fichero local.
    """
    if local_path.exists() and local_path.stat().st_size > 0:
        log(f"URL target ya existe: {local_path} ({local_path.stat().st_size} bytes)")
        return

    local_path.parent.mkdir(parents=True, exist_ok=True)

    log(f"Downloading URL -> file: {url}")
    log(f"Dest: {local_path}")

    with requests.get(
        url,
        stream=True,
        timeout=(10, timeout),
        allow_redirects=True,
        headers=_default_headers()
    ) as r:
        r.raise_for_status()
        _assert_not_html_response(r, url, asset_hint="(expected CSV/binary)")

        tmp = local_path.with_suffix(".part")
        bytes_written = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bytes_written += len(chunk)

        tmp.replace(local_path)

    if not local_path.exists() or local_path.stat().st_size == 0:
        raise RuntimeError(f"Descarga vacía/corrupta desde {url}")

    log(f"Downloaded OK: {local_path} ({local_path.stat().st_size} bytes, wrote={bytes_written})")


def unzip_if_missing(zip_path: Path, out_path: Path):
    """
    Descomprime zip_path en /tmp. Si el .bst viene en subcarpetas, lo mueve a out_path.
    """
    if out_path.exists() and out_path.stat().st_size > 0:
        log(f"BST ya existe: {out_path} ({out_path.stat().st_size} bytes)")
        return

    if not zip_path.exists() or zip_path.stat().st_size == 0:
        raise RuntimeError(f"ZIP inexistente o vacío: {zip_path}")

    log(f"Unzipping: {zip_path} -> {zip_path.parent}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(zip_path.parent)

    try:
        tmp_listing = sorted([p.name for p in zip_path.parent.iterdir()])[:50]
        log(f"/tmp listing (first 50): {tmp_listing}")
    except Exception:
        pass

    if out_path.exists() and out_path.stat().st_size > 0:
        log(f"BST listo tras unzip: {out_path} ({out_path.stat().st_size} bytes)")
        return

    bst_candidates = list(zip_path.parent.rglob("*.bst"))
    if not bst_candidates:
        extracted_any = list(zip_path.parent.rglob("*"))[:30]
        raise RuntimeError(
            f"No hay .bst tras unzip. ZIP={zip_path}. "
            f"Ejemplos extraídos: {[str(x) for x in extracted_any]}"
        )

    bst_candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
    log(f"BST candidates: {[str(p) for p in bst_candidates[:5]]}")
    bst_candidates[0].replace(out_path)

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"No se pudo preparar el BST final en: {out_path}")

    log(f"BST preparado: {out_path} ({out_path.stat().st_size} bytes)")


# -----------------------------
# DESCRIPTIONS
# -----------------------------
def resolve_description_file(project_root: Path) -> Path:
    candidates = [
        project_root / "data" / "processed" / "species_catalog_with_description.csv",
        project_root / "src" / "species_catalog_with_description.csv",
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 0:
            return p

    expected = "\n- " + "\n- ".join(str(x) for x in candidates)
    raise FileNotFoundError("No se encontró archivo de descripciones. Esperaba uno de:" f"{expected}")


def load_descriptions(file_path: Path) -> dict:
    """Carga descripciones desde CSV con el módulo estándar (menos RAM que pandas)."""
    if file_path.suffix.lower() != ".csv":
        raise ValueError(f"Solo se admite CSV para descripciones en Render: {file_path}")

    # Intento 1: coma
    for delimiter in [",", ";"]:
        try:
            with open(file_path, "r", encoding="utf-8", newline="") as f:
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
            # Fallback latin1
            with open(file_path, "r", encoding="latin1", newline="") as f:
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

    raise ValueError(
        f"CSV de descripciones inválido: {file_path}. "
        "Debe tener columnas scientificName y description (separador ',' o ';')."
    )


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

    audio_tf = tf.convert_to_tensor(audio, dtype=tf.float32)
    audio_tf = tf.reshape(audio_tf, [-1])

    scores, embeddings, spectrogram = yamnet(audio_tf)
    emb_mean = tf.reduce_mean(embeddings, axis=0).numpy()
    emb_std = tf.math.reduce_std(embeddings, axis=0).numpy()
    n_frames = int(embeddings.shape[0])
    return np.concatenate([emb_mean, emb_std, [n_frames]])

def predict_top5(booster: xgb.Booster, classes: list[str], x):
    """
    Predice top-5 usando Booster + DMatrix (menor overhead de memoria).
    Requiere que el booster haya sido entrenado con multi:softprob (ideal).
    """
    X = np.asarray([x], dtype=np.float32)
    dm = xgb.DMatrix(X)

    pred = booster.predict(dm)
    # pred puede venir como (n_samples, n_classes) o (n_samples,) si softmax/binary
    pred = np.asarray(pred)
    if pred.ndim == 2:
        proba = pred[0]
    else:
        # si viniera softmax con class index, no tenemos probabilidades -> no sirve para top5
        # y si fuera binary:logistic, pred sería prob de class=1
        raise RuntimeError(
            f"Formato de predicción inesperado: shape={pred.shape}. "
            "Asegúrate de entrenar con objective='multi:softprob' para top-5."
        )

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
    static_folder=str(BASE_DIR / "static"),
)

STATE = {
    "ready": False,
    "booting": False,
    "boot_error": None,
    "boot_step": None,
    "boot_started_at": None,
    "booster": None,          # <-- Booster
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
        log(f"BOOT start. zip={XGB_MODEL_ZIP_PATH} bst={XGB_MODEL_PATH}")

        # 1) Descargar ZIP del modelo desde Releases
        _set_state(step="download_xgb_zip")
        download_release_asset_if_missing(XGB_MODEL_ZIP_PATH, RELEASE_ASSET_NAME)

        # 2) Unzip -> BST
        _set_state(step="unzip_xgb")
        unzip_if_missing(XGB_MODEL_ZIP_PATH, XGB_MODEL_PATH)

        # 3) Cargar modelo con Booster (menos overhead)
        _set_state(step="load_xgb")
        log(f"About to load Booster from {XGB_MODEL_PATH} size={XGB_MODEL_PATH.stat().st_size}")
        booster = xgb.Booster()
        booster.load_model(str(XGB_MODEL_PATH))
        log("Booster loaded OK.")

        # 4) Clases: repo si existe; si no, Releases (evita sklearn/joblib => menos RAM)
        _set_state(step="ensure_classes")
        classes_path = CLASSES_PATH
        if (not classes_path.exists()) or classes_path.stat().st_size == 0:
            log(f"classes.json not found in repo path: {CLASSES_PATH}. Downloading from Releases...")
            download_release_asset_if_missing(CLASSES_TMP_PATH, CLASSES_ASSET_NAME)
            classes_path = CLASSES_TMP_PATH

        if (not classes_path.exists()) or classes_path.stat().st_size == 0:
            raise FileNotFoundError(f"classes.json no encontrado ni descargable: {classes_path}")

        _set_state(step="load_classes")
        with open(classes_path, "r", encoding="utf-8") as f:
            classes = json.load(f)
        if not isinstance(classes, list) or not classes:
            raise ValueError(f"classes.json inválido: {classes_path}")
        log(f"Classes loaded: {classes_path} (n={len(classes)})")

        # 5) Descripciones: DESC_CSV_URL manda
        _set_state(step="download_desc")
        desc_url = (os.getenv("DESC_CSV_URL") or "").strip()
        if desc_url:
            log(f"DESC_CSV_URL set. Downloading descriptions from: {desc_url}")
            download_url_if_missing(DESC_TMP_PATH, desc_url, timeout=60)
            desc_file = DESC_TMP_PATH
        else:
            log("DESC_CSV_URL not set. Resolving description file from repo...")
            desc_file = resolve_description_file(PROJECT_ROOT)

        _set_state(step="load_desc")
        desc_map = load_descriptions(desc_file)
        log(f"Descriptions loaded from: {desc_file} (n={len(desc_map)})")

        _set_state(step="finalize")
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
    Arranque idempotente: si ya está ready o booting, no reinicia.
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

    booster = STATE["booster"]
    classes = STATE["classes"]
    desc_map = STATE["desc_map"]
    desc_file_used = STATE["desc_file"]

    if request.method == "POST":
        f = request.files.get("audio")
        if not f:
            return render_template("index.html", error="No se subió ningún archivo", meta={"desc_file": str(desc_file_used)})

        tmp_path = Path("/tmp") / f"{uuid.uuid4().hex}_{f.filename}"
        f.save(tmp_path)

        try:
            waveform = load_audio(tmp_path)
            x = compute_yamnet_embeddings(waveform)
            top5 = predict_top5(booster, classes, x)

            results = [{
                "scientificName": sp,
                "score": float(score),
                "description": desc_map.get(sp, "No description available."),
            } for sp, score in top5]

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

        return render_template("index.html", results=results, meta={"desc_file": str(desc_file_used)})

    return render_template("index.html", meta={"desc_file": str(desc_file_used)})


if __name__ == "__main__":
    app.run(debug=True)

