## 0. Importar librerías

from __future__ import annotations


import os
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"
import re
import csv
import time
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

import tensorflow as tf
import tensorflow_hub as hub
from pydub import AudioSegment

## 1. Configuración

CSV_URL = "https://raw.githubusercontent.com/anto-rom/Xeno_Canto_Project/main/data/processed/df_final.csv"

PROJECT_ROOT = Path(r"C:\Projects\Xeno_Canto_Project")
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
EMB_DIR = ARTIFACTS_DIR / "embeddings_yamnet"

LOG_DIR = ARTIFACTS_DIR / "logs"
LOG_CSV = LOG_DIR / "embedding_errors.csv"

SAMPLE_RATE = 16000
MIN_AUDIO_SECONDS = 1.0          # descarta audios demasiado cortos
MAX_AUDIO_SECONDS = 30.0         # recorta para consistencia (YAMNet OK con variable, pero esto estabiliza)
REQUEST_TIMEOUT = 45
TRIES_PER_AUDIO = 3
BACKOFF_BASE_SECONDS = 1.5

# Si quieres balancear (cap por especie). Pon None para procesar todo.
MAX_PER_SPECIES = None  # ejemplo: 50

HEADERS = {
    "User-Agent": "XenoCantoProject/1.0 (contact: anto-rom)",
    "Accept": "*/*",
}

CSV_URL = "https://raw.githubusercontent.com/anto-rom/Xeno_Canto_Project/main/data/processed/df_final.csv"

PROJECT_ROOT = Path(r"C:\Projects\Xeno_Canto_Project")
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
EMB_DIR = ARTIFACTS_DIR / "embeddings_yamnet"

LOG_DIR = ARTIFACTS_DIR / "logs"
LOG_CSV = LOG_DIR / "embedding_errors.csv"

SAMPLE_RATE = 16000
MIN_AUDIO_SECONDS = 1.0          # descarta audios demasiado cortos
MAX_AUDIO_SECONDS = 30.0         # recorta para consistencia (YAMNet OK con variable, pero esto estabiliza)
REQUEST_TIMEOUT = 45
TRIES_PER_AUDIO = 3
BACKOFF_BASE_SECONDS = 1.5

# Si quieres balancear (cap por especie). Pon None para procesar todo.
MAX_PER_SPECIES = None  # ejemplo: 50

HEADERS = {
    "User-Agent": "XenoCantoProject/1.0 (contact: anto-rom)",
    "Accept": "*/*",
}

## 2. Checks (ffmpeg + GPU info

def check_ffmpeg() -> None:
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("✅ ffmpeg OK")
    except Exception:
        raise RuntimeError("❌ ffmpeg no está accesible en PATH. Instálalo o añade su ruta al PATH.")

check_ffmpeg()

print("TF version:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))


## 3. Load YAMNet

print("🔄 Cargando YAMNet...")
yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
print("✅ YAMNet cargado")

## 4. Load DataFrame + validaciones

df = pd.read_csv(CSV_URL)

required_cols = {"scientificName", "references", "description"}
missing = required_cols - set(df.columns)
if missing:
    raise KeyError(f"Faltan columnas requeridas: {missing}. Columnas actuales: {df.columns.tolist()}")

print("Rows:", len(df), "Cols:", df.shape[1])
df.head(3)

## 5. Helpers + extracción XC, logging

XC_RE = re.compile(r"(XC\d+)", re.IGNORECASE)

def safe_name(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r'[\\/:*?"<>|]+', "_", s)  # inválidos en Windows
    s = re.sub(r"\s+", " ", s).strip()
    return s[:120] if len(s) > 120 else s  # evita rutas larguísimas

def extract_xc_id(ref_url: str) -> str:
    if ref_url is None or (isinstance(ref_url, float) and pd.isna(ref_url)):
        raise ValueError("references is null")

    ref = str(ref_url).strip()
    m = XC_RE.search(ref)
    if m:
        return m.group(1).upper()

    # fallback: último fragmento numérico si la URL acaba en /123456
    tail = ref.rstrip("/").split("/")[-1]
    if tail.isdigit():
        return f"XC{tail}"

    raise ValueError(f"XC id not found in references: {ref_url}")

def ensure_error_log_header(path: Path) -> None:
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "xc_id", "species", "references", "download_url", "error"])

def log_error(path: Path, xc_id: str, species: str, ref_url: str, download_url: str, err: str) -> None:
    ensure_error_log_header(path)
    ts = pd.Timestamp.utcnow().isoformat()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([ts, xc_id, species, ref_url, download_url, err])

## 6. Descarga streaming con retries + backoff

def download_audio_streaming(download_url: str, timeout: int = REQUEST_TIMEOUT) -> Path:
    """
    Descarga el audio a un archivo temporal con streaming.
    Devuelve el Path del temporal (hay que borrarlo luego).
    """
    last_err = None

    for attempt in range(1, TRIES_PER_AUDIO + 1):
        try:
            with requests.get(download_url, headers=HEADERS, stream=True, timeout=timeout, allow_redirects=True) as r:
                r.raise_for_status()

                # Guardar a temporal
                with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp:
                    for chunk in r.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            tmp.write(chunk)
                    tmp_path = Path(tmp.name)

            # sanity mínimo: evita HTML/página de error “pesada”
            if tmp_path.exists() and tmp_path.stat().st_size > 20_000:
                return tmp_path

            tmp_path.unlink(missing_ok=True)
            raise RuntimeError("Downloaded file too small (likely error page).")

        except Exception as e:
            last_err = e
            if attempt < TRIES_PER_AUDIO:
                time.sleep(BACKOFF_BASE_SECONDS * attempt)
            else:
                break

    raise RuntimeError(f"Download failed after retries: {last_err}")

## 7. Audio - waveform float32 (16k mono) + recorte + normalización

def audio_file_to_waveform(
    audio_path: Path,
    sample_rate: int = SAMPLE_RATE,
    min_seconds: float = MIN_AUDIO_SECONDS,
    max_seconds: float = MAX_AUDIO_SECONDS,
) -> np.ndarray:
    """
    Carga cualquier audio soportado por ffmpeg/pydub y devuelve waveform float32 en [-1,1]
    """
    audio = AudioSegment.from_file(str(audio_path))
    audio = audio.set_frame_rate(sample_rate).set_channels(1)

    # recorte a max_seconds para estabilidad
    if max_seconds is not None and max_seconds > 0:
        audio = audio[: int(max_seconds * 1000)]

    duration_sec = len(audio) / 1000.0
    if duration_sec < min_seconds:
        raise ValueError(f"Audio too short: {duration_sec:.2f}s")

    # samples int -> float32
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)

    # normalización según sample_width (8/16/24/32 bits)
    max_val = float(1 << (8 * audio.sample_width - 1))
    if max_val <= 0:
        raise ValueError("Invalid sample width for normalization")

    samples /= max_val

    # clamp por seguridad
    samples = np.clip(samples, -1.0, 1.0)

    return samples

## 8. Embeddings YAMNet + pooling + guardado

def compute_yamnet_embeddings(waveform: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Devuelve: emb_frames(float16), emb_mean(float16), emb_std(float16)
    """
    wave = tf.convert_to_tensor(waveform, dtype=tf.float32)
    scores, embeddings, spectrogram = yamnet(wave)

    emb = embeddings.numpy().astype(np.float16)  # (T, 1024)
    emb_mean = emb.mean(axis=0).astype(np.float16)
    emb_std = emb.std(axis=0).astype(np.float16)
    return emb, emb_mean, emb_std

def save_npz(out_file: Path, species: str, xc_id: str, emb: np.ndarray, emb_mean: np.ndarray, emb_std: np.ndarray) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_file,
        embeddings=emb,
        emb_mean=emb_mean,
        emb_std=emb_std,
        species=species,
        xc_id=xc_id
    )

## 9. Balanceo por especie

def apply_cap_per_species(df: pd.DataFrame, max_per_species: Optional[int]) -> pd.DataFrame:
    if max_per_species is None:
        return df
    return (
        df.groupby("scientificName", group_keys=False)
          .head(max_per_species)
          .reset_index(drop=True)
    )

df_run = apply_cap_per_species(df, MAX_PER_SPECIES)
print("Rows to process:", len(df_run))

## 10. Main loop (reanudable)

# --- Nos aseguramos de que el directorio de logs exista ---
LOG_DIR.mkdir(parents=True, exist_ok=True)

if not LOG_CSV.exists():
    with open(LOG_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "xc_id", "species", "references", "download_url", "error"])

ensure_error_log_header(LOG_CSV)

processed = 0
skipped = 0
failed = 0

for _, row in tqdm(df_run.iterrows(), total=len(df_run)):
    species = str(row["scientificName"]).strip()
    ref_url = row["references"]

    try:
        xc_id = extract_xc_id(ref_url)
        xc_num = xc_id.replace("XC", "")  # "123456"

        species_dir = EMB_DIR / safe_name(species)
        out_file = species_dir / f"{xc_id}.npz"

        if out_file.exists():
            skipped += 1
            continue

        download_url = f"https://xeno-canto.org/{xc_num}/download"

        tmp_path = None
        try:
            tmp_path = download_audio_streaming(download_url)
            waveform = audio_file_to_waveform(tmp_path)
            emb, emb_mean, emb_std = compute_yamnet_embeddings(waveform)
            save_npz(out_file, species, xc_id, emb, emb_mean, emb_std)
            processed += 1

        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

    except Exception as e:
        failed += 1
        # best effort para loggear xc_id aunque haya fallado antes
        try:
            xc_id_for_log = extract_xc_id(ref_url)
            xc_num = xc_id_for_log.replace("XC", "")
            download_url_for_log = f"https://xeno-canto.org/{xc_num}/download"
        except Exception:
            xc_id_for_log = ""
            download_url_for_log = ""

        log_error(
            LOG_CSV,
            xc_id=xc_id_for_log,
            species=species,
            ref_url=str(ref_url),
            download_url=download_url_for_log,
            err=str(e)
        )

print(f"✅ processed={processed} | ⏭️ skipped={skipped} | ❌ failed={failed}")
print("Error log:", LOG_CSV)

## 11. KPI rápido

# Cuenta embeddings generados
n_files = sum(1 for _ in EMB_DIR.rglob("*.npz"))
print("Total embeddings files:", n_files)

# Últimos errores (si los hay)
if LOG_CSV.exists():
    err_df = pd.read_csv(LOG_CSV)
    print("Errors logged:", len(err_df))
    display(err_df.tail(10))
