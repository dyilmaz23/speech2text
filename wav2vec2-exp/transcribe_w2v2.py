import os
import time
import csv
from pathlib import Path

import argparse

import soundfile as sf
import torch
import torchaudio


import torch
import torchaudio
from transformers import AutoProcessor, AutoModelForCTC

# ----------------- AYARLAR (tr/en arası değişken) -----------------
MODEL_NAME = "m3hrdadfi/wav2vec2-large-xlsr-turkish"  # EN baseline
AUDIO_DIR = "data/data-tr"

OUT_DIR = "outputs/tr"
TRANSCRIPT_DIR = os.path.join(OUT_DIR, "transcripts")
LOG_DIR = os.path.join(OUT_DIR, "logs")

CSV_LOG = os.path.join(LOG_DIR, "wav2vec2_latency_log.csv")

EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
# -------------------------------------------

def find_audio_files(root: str):
    root_path = Path(root)
    files = [p for p in root_path.rglob("*") if p.is_file() and p.suffix.lower() in EXTS]
    return sorted(files)

def load_audio_mono_16k(path):
    wav, sr = sf.read(str(path), always_2d=False)  # numpy
    if wav.ndim > 1:  # stereo -> mono
        wav = wav.mean(axis=1)
    wav = torch.tensor(wav, dtype=torch.float32)

    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
        sr = 16000
    return wav, sr

def safe_stem(p):
    # Dosya adını output için güvenli hale getirir
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in p.stem)


@torch.inference_mode()
def transcribe_one(path: Path, processor, model, device):
    audio, sr = load_audio_mono_16k(path)

    t0 = time.perf_counter()
    inputs = processor(audio.numpy(), sampling_rate=sr, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)

    logits = model(input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)
    text = processor.batch_decode(pred_ids)[0].strip()
    t1 = time.perf_counter()

    latency_sec = t1 - t0
    duration_sec = audio.shape[0] / sr

    return text, latency_sec, duration_sec

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", default="data/data-tr")
    parser.add_argument("--out_dir", default="outputs/tr")
    parser.add_argument("--model", default="m3hrdadfi/wav2vec2-large-xlsr-turkish")
    args = parser.parse_args()

    global MODEL_NAME, AUDIO_DIR, OUT_DIR, TRANSCRIPT_DIR, LOG_DIR, CSV_LOG
    MODEL_NAME = args.model
    AUDIO_DIR = args.audio_dir
    OUT_DIR = args.out_dir
    TRANSCRIPT_DIR = os.path.join(OUT_DIR, "transcripts")
    LOG_DIR = os.path.join(OUT_DIR, "logs")
    CSV_LOG = os.path.join(LOG_DIR, "latency_log.csv")

    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model:  {MODEL_NAME}")

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForCTC.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    files = find_audio_files(AUDIO_DIR)
    if not files:
        raise SystemExit(f"No audio files found under: {AUDIO_DIR}")

    # CSV header (append-safe)
    write_header = not os.path.exists(CSV_LOG)
    with open(CSV_LOG, "a", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        if write_header:
            writer.writerow([
                "file_path",
                "audio_duration_sec",
                "inference_latency_sec",
                "rtf",  # real-time factor = latency / duration
                "model_name"
            ])

        for i, path in enumerate(files, start=1):
            print(f"\n[{i}/{len(files)}] {path.name}")
            text, latency, dur = transcribe_one(path, processor, model, device)
            rtf = (latency / dur) if dur > 0 else None

            out_name = safe_stem(path) + ".txt"
            out_path = os.path.join(TRANSCRIPT_DIR, out_name)

            with open(out_path, "w", encoding="utf-8") as fout:
                fout.write(text + "\n")

            writer.writerow([str(path), f"{dur:.3f}", f"{latency:.3f}", f"{rtf:.4f}", MODEL_NAME])

            print(f"Saved transcript -> {out_path}")
            print(f"Latency: {latency:.3f}s | Audio: {dur:.3f}s | RTF: {rtf:.4f}")

    print("\nDONE.")
    print(f"Log CSV -> {CSV_LOG}")

if __name__ == "__main__":
    main()
