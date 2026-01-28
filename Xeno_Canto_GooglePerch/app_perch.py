import os
import time
import warnings
import numpy as np
import librosa
import joblib
import gradio as gr
import tensorflow as tf
import tensorflow_hub as hub

# --- CONFIGURACIÓN DE ALERTAS Y CPU ---
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Bloquea GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Silencia warnings TF
warnings.filterwarnings('ignore')

# --- RUTAS DE ARCHIVOS ---
CARPETA_MODELO = "/Users/gabrielajara/anaconda_projects/Sonidos_Aves_del_mundo_/perch_model"
PIPELINE_FILE = "perch_logreg_softmax.joblib"
LABEL_ENCODER_FILE = "label_encoder.joblib"

# --- CLASE PRINCIPAL ---
class BirdClassifierM2:
    def __init__(self, pipeline_path, le_path, model_path):
        print("🚀 Iniciando motor PERCH Pro (1280 dim) en CPU-only...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ No se encontró la carpeta del modelo: {model_path}")

        with tf.device('/CPU:0'):
            self.perch_model = hub.load(model_path)
            print("✅ Modelo PERCH cargado correctamente.")

        if not os.path.exists(pipeline_path):
            raise FileNotFoundError(f"❌ No se encontró el pipeline: {pipeline_path}")
        if not os.path.exists(le_path):
            raise FileNotFoundError(f"❌ No se encontró el label encoder: {le_path}")

        self.pipeline = joblib.load(pipeline_path)
        self.label_encoder = joblib.load(le_path)
        print("✅ Clasificadores Softmax y Label Encoder listos.")

    def process_audio_multi_window(self, audio_path):
        audio, _ = librosa.load(audio_path, sr=32000, mono=True)

        window_size = 160000
        audio_len = len(audio)

        if audio_len < window_size:
            audio = np.pad(audio, (0, window_size - audio_len), mode='constant')
            audio_chunks = [audio]
        else:
            audio_chunks = [
                audio[i:i + window_size]
                for i in range(0, audio_len - window_size + 1, window_size)
            ]

        all_embeddings = []
        with tf.device('/CPU:0'):
            for chunk in audio_chunks:
                tensor = tf.convert_to_tensor(chunk[np.newaxis, :], dtype=tf.float32)
                outputs = self.perch_model.infer_tf(tensor)

                if isinstance(outputs, dict):
                    emb = outputs['embedded_feature']
                else:
                    emb = outputs[1] if isinstance(outputs, (list, tuple)) else outputs

                all_embeddings.append(np.mean(emb, axis=0))

        final_embedding = np.mean(all_embeddings, axis=0).reshape(1, -1)
        return final_embedding

    def predict(self, audio_path):
        if audio_path is None:
            return "Por favor, sube un archivo de audio."

        try:
            start_t = time.time()
            features = self.process_audio_multi_window(audio_path)

            probs = self.pipeline.predict_proba(features)
            confianza = np.max(probs)

            if confianza < 0.20:
                return (
                    "❓ Especie no identificada con claridad\n"
                    f"📊 Confianza insuficiente: {confianza:.2%}\n"
                    "📝 Nota: El sonido es muy débil o la especie no está en el entrenamiento."
                )

            pred_numeric = self.pipeline.predict(features)
            especie_nombre = self.label_encoder.inverse_transform(pred_numeric)
            nombre = especie_nombre[0]
            duration = time.time() - start_t

            return (
                f"🐦 Especie: {nombre}\n"
                f"📊 Confianza: {confianza:.2%}\n"
                f"⚡ Tiempo CPU M2 (Multi-ventana): {duration:.2f}s"
            )

        except Exception as e:
            return f"❌ Error en el análisis: {str(e)}"

# --- LANZAMIENTO DE INTERFAZ ---
if __name__ == "__main__":
    try:
        predictor = BirdClassifierM2(
            PIPELINE_FILE,
            LABEL_ENCODER_FILE,
            CARPETA_MODELO
        )

        app = gr.Interface(
            fn=predictor.predict,
            inputs=gr.Audio(type="filepath", label="Cargar Canto (Cualquier duración)"),
            outputs=gr.Textbox(label="Identificación Inteligente"),
            title="BirdID Ecuador Pro - Apple Silicon M2",
            description="Analiza el audio completo en ventanas de 5s para máxima precisión (PERCH 1280dim).",
            theme=gr.themes.Soft()
        )

        print("✅ Iniciando interfaz con link público...")
        app.launch(inline=True, share=True)

    except Exception as e:
        print(f"❌ Error al iniciar: {e}")
