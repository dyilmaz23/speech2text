import os
import time
import csv
import argparse
from pathlib import Path

import whisper

EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


def find_audio_files(root: str):
    root_path = Path(root)
    files = [p for p in root_path.rglob("*") if p.is_file() and p.suffix.lower() in EXTS]
    return sorted(files)


def safe_stem(p: Path) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in p.stem)


def approx_duration_from_segments(result: dict) -> float:
    """
    torchaudio/soundfile bağımlılığı olmadan duration.
    Whisper'ın segment zamanlarından yaklaşık duration hesaplarız.
    """
    segs = result.get("segments") or []
    if not segs:
        return 0.0
    # en sona kadar olan zamanı duration gibi alıyoruz
    return float(segs[-1].get("end", 0.0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium", help="Whisper model: tiny/base/small/medium/large-v3 ...")
    parser.add_argument("--audio_dir", required=True, help="Input audio directory (e.g., data/data-en)")
    parser.add_argument("--out_dir", required=True, help="Output directory (e.g., outputs/en_medium)")
    parser.add_argument("--language", required=True, help="Fixed decoding language: en or tr")
    parser.add_argument("--task", default="transcribe", choices=["transcribe", "translate"], help="Whisper task")
    args = parser.parse_args()

    transcript_dir = os.path.join(args.out_dir, "transcripts")
    log_dir = os.path.join(args.out_dir, "logs")
    os.makedirs(transcript_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    device = "cuda" if whisper.torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model:  {args.model}")
    print(f"Lang:   {args.language} | Task: {args.task}")
    print(f"Input:  {args.audio_dir}")
    print(f"Output: {args.out_dir}")

    model = whisper.load_model(args.model, device=device)

    files = find_audio_files(args.audio_dir)
    if not files:
        raise SystemExit(f"No audio files found under: {args.audio_dir}")

    csv_log = os.path.join(log_dir, f"whisper_{args.model}_{args.language}_latency_log.csv")
    write_header = not os.path.exists(csv_log)

    with open(csv_log, "a", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        if write_header:
            writer.writerow([
                "file_path",
                "audio_duration_sec_est",
                "inference_latency_sec",
                "rtf_est",
                "model_name",
                "language",
                "task",
                "device",
            ])

        for i, path in enumerate(files, start=1):
            print(f"\n[{i}/{len(files)}] {path.name}")

            t0 = time.perf_counter()
            result = model.transcribe(
                str(path),
                task=args.task,
                language=args.language,  # fixed language (senin istediğin)
                fp16=False,              # CPU uyumlu
                verbose=False
            )
            t1 = time.perf_counter()

            text = (result.get("text") or "").strip()
            latency_sec = t1 - t0
            duration_est = approx_duration_from_segments(result)
            rtf_est = (latency_sec / duration_est) if duration_est > 0 else None

            out_path = os.path.join(transcript_dir, safe_stem(path) + ".txt")
            with open(out_path, "w", encoding="utf-8") as fout:
                fout.write(text + "\n")

            writer.writerow([
                str(path),
                f"{duration_est:.3f}",
                f"{latency_sec:.3f}",
                f"{rtf_est:.4f}" if rtf_est is not None else "",
                args.model,
                args.language,
                args.task,
                device
            ])

            if rtf_est is not None:
                print(f"Saved transcript -> {out_path}")
                print(f"Latency: {latency_sec:.3f}s | Audio(est): {duration_est:.3f}s | RTF(est): {rtf_est:.4f}")
            else:
                print(f"Saved transcript -> {out_path}")
                print(f"Latency: {latency_sec:.3f}s | Audio(est): {duration_est:.3f}s")

    print("\nDONE.")
    print(f"Log CSV -> {csv_log}")


if __name__ == "__main__":
    main()
