import os
import json
import time
import argparse
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI

# OpenAI Responses API is recommended for new projects.
# https://platform.openai.com/docs/api-reference/responses/create
# https://platform.openai.com/docs/guides/text
# (see citations in report, not in code)

def read_text(p: Path) -> str:
    # Be tolerant to encoding quirks.
    return p.read_text(encoding="utf-8", errors="replace").strip()

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def run_one(client: OpenAI, model: str, system_prompt: str, transcript: str) -> dict:
    """
    Returns a dict with: output_text, latency_s, raw_response_id (and usage if available).
    """
    t0 = time.time()
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript},
        ],
        # Keep deterministic for evaluation comparability:
        temperature=0.2,
        max_output_tokens=3000,
    )
    latency_s = time.time() - t0

    # Python SDK convenience: aggregated text output
    output_text = getattr(resp, "output_text", None) or ""

    out = {
        "model": model,
        "response_id": getattr(resp, "id", None),
        "latency_s": latency_s,
        "output_text": output_text,
    }

    # Usage fields can vary by SDK/version; keep it optional.
    usage = getattr(resp, "usage", None)
    if usage is not None:
        try:
            out["usage"] = usage.model_dump()  # pydantic-like
        except Exception:
            out["usage"] = str(usage)

    return out

def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not found. Put it in .env or export it.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=["en", "tr"], required=True)
    parser.add_argument("--transcripts_dir", required=True, help="Folder containing .txt transcripts (Whisper medium outputs).")
    parser.add_argument("--out_root", default="semantic-exp/outputs")
    parser.add_argument("--log_dir", default="semantic-exp/logs")
    parser.add_argument("--prompt_path", required=True)
    parser.add_argument("--models", nargs="+", default=["gpt-4o-mini", "gpt-4.1"])
    args = parser.parse_args()

    transcripts_dir = Path(args.transcripts_dir)
    out_root = Path(args.out_root) / args.lang
    log_dir = Path(args.log_dir)
    prompt_path = Path(args.prompt_path)

    system_prompt = read_text(prompt_path)

    ensure_dir(out_root)
    ensure_dir(log_dir)

    client = OpenAI(api_key=api_key)

    files = sorted([p for p in transcripts_dir.glob("*.txt") if p.is_file()])
    if not files:
        raise SystemExit(f"No .txt files found in {transcripts_dir}")

    # Log file (JSONL)
    log_path = log_dir / f"semantic_{args.lang}_{int(time.time())}.jsonl"

    for model in args.models:
        ensure_dir(out_root / model)

    with log_path.open("w", encoding="utf-8") as log_f:
        for fp in tqdm(files, desc=f"Semantic ({args.lang})"):
            transcript = read_text(fp)

            for model in args.models:
                result = run_one(client, model=model, system_prompt=system_prompt, transcript=transcript)

                # Save plain text output
                out_txt = out_root / model / fp.name
                out_txt.write_text(result["output_text"], encoding="utf-8")

                # Log JSON line
                log_item = {
                    "lang": args.lang,
                    "file": fp.name,
                    "transcripts_dir": str(transcripts_dir),
                    **result,
                }
                log_f.write(json.dumps(log_item, ensure_ascii=False) + "\n")

    print(f"\nSaved outputs under: {out_root}")
    print(f"Saved log: {log_path}")

if __name__ == "__main__":
    main()
