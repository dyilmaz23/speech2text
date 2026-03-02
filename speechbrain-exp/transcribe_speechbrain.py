import os
import time
import csv
import argparse
from pathlib import Path

import torch
import torchaudio

import torchaudio as _ta
# --- Compatibility shim for newer torchaudio versions (SpeechBrain import-time check) ---
if not hasattr(_ta, "list_audio_backends"):
    def _list_audio_backends():
        return []
    _ta.list_audio_backends = _list_audio_backends
# -------------------------------------------------------------------------------

from speechbrain.inference import EncoderDecoderASR

EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


def find_audio_files(root: str):
    root_path = Path(root)
    files = [p for p in root_path.rglob("*") if p.is_file() and p.suffix.lower() in EXTS]
    return sorted(files)


def safe_stem(p: Path) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in p.stem)


def load_audio_mono_16k(path: Path):
    wav, sr = torchaudio.load(str(path))  # [channels, time]
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
        sr = 16000
    wav = wav.squeeze(0)  # [time]
    duration_sec = wav.shape[0] / float(sr) if sr > 0 else 0.0
    return wav, sr, duration_sec


def chunk_audio(wav_1d: torch.Tensor, sr: int, chunk_sec: float, overlap_sec: float):
    """
    Returns list of 1D chunks (torch.Tensor) with overlap.
    """
    chunk_len = int(chunk_sec * sr)
    hop = int((chunk_sec - overlap_sec) * sr)
    if hop <= 0:
        raise ValueError("overlap_sec must be smaller than chunk_sec")

    n = wav_1d.shape[0]
    chunks = []
    start = 0
    while start < n:
        end = min(start + chunk_len, n)
        chunk = wav_1d[start:end]
        # çok kısa chunk'lar bazen sorun çıkarabiliyor; 0.5s altını atla
        if chunk.numel() >= int(0.5 * sr):
            chunks.append(chunk)
        if end == n:
            break
        start += hop
    return chunks


@torch.inference_mode()
def transcribe_long_audio(asr_model, wav_1d: torch.Tensor, sr: int, chunk_sec: float, overlap_sec: float):
    """
    Chunk-by-chunk transcription for long audios.
    Returns concatenated text and total inference latency (sec).
    """
    chunks = chunk_audio(wav_1d, sr, chunk_sec, overlap_sec)

    texts = []
    total_latency = 0.0

    for c in chunks:
        wav_batch = c.unsqueeze(0).to(asr_model.device)
        # SpeechBrain expects wav_lens as relative lengths in [0,1]
        wav_lens = torch.tensor([1.0], device=asr_model.device)

        t0 = time.perf_counter()
        out = asr_model.transcribe_batch(wav_batch, wav_lens)
        t1 = time.perf_counter()

        total_latency += (t1 - t0)

        # out can be list/tuple or a tensor-like; handle safely
        if isinstance(out, (list, tuple)):
            txt = str(out[0]).strip()
        else:
            txt = str(out).strip()

        if txt:
            texts.append(txt)

    # Basit bir join; overlap varsa tekrarlar olabilir ama değerlendirmede kabul edilebilir.
    # İstersen sonra "deduplicate overlap" iyileştirmesi yaparız.
    return " ".join(texts).strip(), total_latency


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", required=True, help="Input audio directory (e.g., data/data-en)")
    parser.add_argument("--out_dir", required=True, help="Output directory (e.g., outputs/en)")
    parser.add_argument(
        "--model_source",
        default="speechbrain/asr-transformer-transformerlm-librispeech",
        help="HuggingFace SpeechBrain model id",
    )
    parser.add_argument(
        "--savedir",
        default="pretrained_models/speechbrain_asr_transformer_librispeech",
        help="Local cache dir for downloaded model",
    )
    parser.add_argument("--chunk_sec", type=float, default=25.0, help="Chunk length in seconds (for long audio).")
    parser.add_argument("--overlap_sec", type=float, default=2.0, help="Chunk overlap in seconds.")
    args = parser.parse_args()

    transcript_dir = os.path.join(args.out_dir, "transcripts")
    log_dir = os.path.join(args.out_dir, "logs")
    os.makedirs(transcript_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"SpeechBrain model: {args.model_source}")
    print(f"Input:  {args.audio_dir}")
    print(f"Output: {args.out_dir}")
    print(f"Chunking: {args.chunk_sec}s with {args.overlap_sec}s overlap")

    asr_model = EncoderDecoderASR.from_hparams(
        source=args.model_source,
        savedir=args.savedir,
        run_opts={"device": device},
    )

    files = find_audio_files(args.audio_dir)
    if not files:
        raise SystemExit(f"No audio files found under: {args.audio_dir}")

    csv_log = os.path.join(log_dir, "speechbrain_latency_log.csv")
    write_header = not os.path.exists(csv_log)

    with open(csv_log, "a", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        if write_header:
            writer.writerow([
                "file_path",
                "audio_duration_sec",
                "inference_latency_sec",
                "rtf",
                "chunk_sec",
                "overlap_sec",
                "model_source",
                "device",
            ])

        for i, path in enumerate(files, start=1):
            print(f"\n[{i}/{len(files)}] {path.name}")

            wav, sr, duration_sec = load_audio_mono_16k(path)

            text, latency_sec = transcribe_long_audio(
                asr_model, wav, sr, args.chunk_sec, args.overlap_sec
            )
            rtf = (latency_sec / duration_sec) if duration_sec > 0 else None

            out_path = os.path.join(transcript_dir, safe_stem(path) + ".txt")
            with open(out_path, "w", encoding="utf-8") as fout:
                fout.write(text + "\n")

            writer.writerow([
                str(path),
                f"{duration_sec:.3f}",
                f"{latency_sec:.3f}",
                f"{rtf:.4f}" if rtf is not None else "",
                args.chunk_sec,
                args.overlap_sec,
                args.model_source,
                device,
            ])

            print(f"Saved transcript -> {out_path}")
            if rtf is not None:
                print(f"Latency: {latency_sec:.3f}s | Audio: {duration_sec:.3f}s | RTF: {rtf:.4f}")
            else:
                print(f"Latency: {latency_sec:.3f}s | Audio: {duration_sec:.3f}s")

    print("\nDONE.")
    print(f"Log CSV -> {csv_log}")


if __name__ == "__main__":
    main()
