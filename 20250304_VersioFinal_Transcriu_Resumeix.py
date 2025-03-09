import os
import librosa
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
from moviepy.editor import VideoFileClip
import whisper
import traceback
from transformers import pipeline

# ================= CONFIGURACIONES =================
MAX_SEGMENT_DURATION = 60          # Duración máxima de segmento (segundos)
MAX_TOKENS = 250                   # Máximo número de tokens para dividir el texto del SRT
SEGMENT_DURATION = 20              # Duración para análisis preciso
MAX_TRANSCRIPTION_DURATION = 15    # Máxima duración de transcripción
MODEL_NAME = "facebook/bart-large-cnn"  # Modelo de Hugging Face para resumen

# Diccionarios de modelos de traducción
# Para traducir desde el idioma de la transcripción a español:
REVERSE_TRANSLATION_MODELS = {
    "ca": "Helsinki-NLP/opus-mt-ca-es",
    "en": "Helsinki-NLP/opus-mt-en-es",
    "fr": "Helsinki-NLP/opus-mt-fr-es",
    "de": "Helsinki-NLP/opus-mt-de-es",
    "ar": "Helsinki-NLP/opus-mt-ar-es",
    "zh": "Helsinki-NLP/opus-mt-zh-es"
}
# Para traducir desde español al idioma de la transcripción:
FORWARD_TRANSLATION_MODELS = {
    "ca": "Helsinki-NLP/opus-mt-es-ca",
    "en": "Helsinki-NLP/opus-mt-es-en",
    "fr": "Helsinki-NLP/opus-mt-es-fr",
    "de": "Helsinki-NLP/opus-mt-es-de",
    "ar": "Helsinki-NLP/opus-mt-es-ar",
    "zh": "Helsinki-NLP/opus-mt-es-zh"
}

# Global cache para pipelines de resumen y traducción
summarizer_es = None
reverse_translators = {}
forward_translators = {}

# ====================================================
# =============== FUNCIONES AUXILIARES ===============
# ====================================================
def translate_large_text(text, translator, chunk_size=100, max_length=512):
    """
    Divide 'text' en trozos de 'chunk_size' palabras y traduce cada trozo por separado
    para evitar el error de secuencia demasiado larga. Ajusta chunk_size si persiste el error.
    """
    tokens = text.split()
    translated_chunks = []

    for i in range(0, len(tokens), chunk_size):
        sub_text = " ".join(tokens[i:i + chunk_size])
        try:
            # max_length=512 evita sobrepasar el límite del modelo.  
            # Si aun así falla, reduce chunk_size más o usa truncation=True (con posible pérdida de texto).
            out = translator(sub_text, max_length=max_length)
            if out and 'translation_text' in out[0]:
                translated_chunks.append(out[0]['translation_text'])
            else:
                translated_chunks.append("[ERROR AL TRADUCIR ESTE FRAGMENTO]")
        except Exception as e:
            print(f"Error al traducir fragmento: {e}")
            translated_chunks.append("[ERROR AL TRADUCIR ESTE FRAGMENTO]")

    return " ".join(translated_chunks)

# ================= FUNCIONES DE TRANSCRIPCIÓN =================
def extract_audio_from_video(video_filename, output_audio_filename="temp_audio.wav"):
    try:
        print(f"Extrayendo audio de {video_filename}...")
        clip = VideoFileClip(video_filename)
        clip.audio.write_audiofile(output_audio_filename, codec='pcm_s16le')
        print(f"Audio extraído correctamente: {output_audio_filename}")
        return output_audio_filename
    except Exception as e:
        print(f"Error al extraer audio: {e}")
        return None

def detect_voice_segments(audio_filename):
    try:
        print(f"Detectando segmentos de voz en {audio_filename}...")
        y, sample_rate = librosa.load(audio_filename, sr=16000)
        intervals = librosa.effects.split(y, top_db=25)
        segments = [(interval[0] / sample_rate, interval[1] / sample_rate) for interval in intervals]
        print(f"Segmentos detectados: {len(segments)}")
        return segments, sample_rate
    except Exception as e:
        print(f"Error al detectar segmentos de voz: {e}")
        return [], 16000

def transcribe_audio_segment_with_whisper(audio_filename, start, end, model, sample_rate, language="ca"):
    try:
        print(f"Transcribiendo segmento {start:.2f}-{end:.2f} en '{language}'...")
        audio = whisper.load_audio(audio_filename)
        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)
        audio_segment = audio[start_sample:end_sample]
        result = model.transcribe(audio_segment, language=language)
        return result.get('text', "[Inaudible]")
    except Exception as e:
        print(f"Error al transcribir {start:.2f}-{end:.2f}: {e}")
        return "[Error]"

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def generate_srt_file(results, srt_filename):
    with open(srt_filename, "w", encoding="utf-8") as srt_file:
        for i, (time_in, time_out, text) in enumerate(results):
            srt_file.write(f"{i + 1}\n{time_in} --> {time_out}\n{text}\n\n")
    print(f"Archivo SRT guardado: {srt_filename}")

def process_audio(video_path, transcription_language="ca"):
    audio_filename = extract_audio_from_video(video_path)
    if not audio_filename:
        messagebox.showerror("Error", "No se pudo extraer el audio del video")
        return

    srt_filename = os.path.splitext(video_path)[0] + ".srt"
    model = whisper.load_model("small")
    segments, sample_rate = detect_voice_segments(audio_filename)
    if not segments:
        messagebox.showerror("Error", "No se detectaron segmentos de voz en el audio.")
        return

    results = []
    for start, end in segments:
        text = transcribe_audio_segment_with_whisper(audio_filename, start, end, model, sample_rate, language=transcription_language)
        results.append((format_time(start), format_time(end), text))

    generate_srt_file(results, srt_filename)
    messagebox.showinfo("Éxito", f"Subtítulos generados: {srt_filename}")

# ================= FUNCIONES DE RESUMEN =================
def extract_text_from_srt(srt_path):
    """ Extrae solo el texto de un archivo SRT, eliminando números y marcas de tiempo. """
    with open(srt_path, "r", encoding="utf-8") as file:
        srt_text = file.read()

    # Filtrar solo el texto
    lines = srt_text.split("\n")
    extracted_text = "\n".join(line.strip() for line in lines if not line.strip().isdigit() and "-->" not in line and line.strip())

    if not extracted_text.strip():
        print("ERROR: No se extrajo texto del SRT.")
        return None

    return extracted_text

def split_text(text, max_tokens=MAX_TOKENS):
    """ Divide el texto en fragmentos más pequeños si excede el límite de tokens. """
    words = text.split()
    segments = [words[i:i + max_tokens] for i in range(0, len(words), max_tokens)]
    return [" ".join(segment) for segment in segments]

def generate_summary(srt_path):
    """ Genera un resumen del texto extraído del archivo SRT. """
    text = extract_text_from_srt(srt_path)
    if text is None:
        return "No se pudo extraer texto del SRT."

    segments = split_text(text)
    summarizer = pipeline("summarization", model=MODEL_NAME)

    summaries = []
    try:
        for i, segment in enumerate(segments):
            print(f"Resumiendo fragmento {i + 1}/{len(segments)}...")
            try:
                summary = summarizer(segment, max_length=100, min_length=30, do_sample=False)
                summaries.append(summary[0]['summary_text'])
            except Exception as e:
                print(f"Error en fragmento {i + 1}: {e}")
                summaries.append("[ERROR AL RESUMIR ESTE FRAGMENTO]")

        return " ".join(summaries)
    
    except Exception as e:
        print(f"Error al resumir: {e}")
        return "Error al generar resumen."

def process_summary(transcription_language):
    srt_filename = filedialog.askopenfilename(title="Seleccionar archivo SRT", filetypes=[("Archivos SRT", "*.srt")])
    if not srt_filename:
        messagebox.showwarning("Advertencia", "Seleccione un archivo SRT primero")
        return

    summary_filename = os.path.splitext(srt_filename)[0] + "_summary.txt"

    def generate_summary_thread():
        summary = generate_summary(srt_filename)  
        if "No se pudo extraer texto del SRT" in summary or "[ERROR" in summary:
            messagebox.showerror("Error", summary)
        else:
            with open(summary_filename, "w", encoding="utf-8") as summary_file:
                summary_file.write(summary)
            messagebox.showinfo("Resumen generado", f"Resumen guardado en:\n{summary_filename}")

    threading.Thread(target=generate_summary_thread, daemon=True).start()



# ================= INTERFAZ GRÁFICA (TKINTER) =================
def main_gui():
    root = tk.Tk()
    root.title("Transcripción y Resumen de Video")
    root.geometry("500x400")

    # Variable para la ruta del video y el idioma de transcripción (y resumen)
    video_path = tk.StringVar()
    transcription_lang = tk.StringVar(value="ca")  # Ejemplo: "ca" para catalán, "es" para castellano, etc.

    # --- Sección Transcripción ---
    transcription_frame = tk.LabelFrame(root, text="Transcripción", padx=10, pady=10)
    transcription_frame.pack(padx=10, pady=10, fill="both", expand=True)
    
    tk.Button(transcription_frame, text="Seleccionar Video", 
              command=lambda: video_path.set(filedialog.askopenfilename(title="Seleccionar archivo de video",
                                                                        filetypes=[("Archivos de video", "*.mp4 *.avi *.mov *.mkv")] ))
             ).pack(pady=5)
    tk.Label(transcription_frame, textvariable=video_path).pack(pady=5)
    
    tk.Label(transcription_frame, text="Idioma de transcripción (código: 'ca'- Catalán, 'es'- Español, 'en' - Inglés, 'fr' - Francés, 'de' - Alemán, 'ar' - Árabe, 'zh'- Chino):").pack(pady=5)
    tk.Entry(transcription_frame, textvariable=transcription_lang).pack(pady=5)
    
    tk.Button(transcription_frame, text="Generar Subtítulos", 
              command=lambda: threading.Thread(target=process_audio, args=(video_path.get(), transcription_lang.get()), daemon=True).start()
             ).pack(pady=10)
    
    # --- Sección Resumen ---
    summary_frame = tk.LabelFrame(root, text="Resumen", padx=10, pady=10)
    summary_frame.pack(padx=10, pady=10, fill="both", expand=True)
    
    tk.Label(summary_frame, text="El resumen se realizará en el mismo idioma que la transcripción.").pack(pady=5)
    tk.Button(summary_frame, text="Generar Resumen desde SRT", 
              command=lambda: process_summary(transcription_lang.get())
             ).pack(pady=10)
    
    root.mainloop()

if __name__ == "__main__":
    main_gui()
