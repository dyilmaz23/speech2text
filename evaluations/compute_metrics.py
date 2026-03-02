import argparse
import os
import glob
import re
import unicodedata
import pandas as pd
from jiwer import wer, cer


# --- Normalization helpers ---
_punct_re = re.compile(r"[^\w\s]", flags=re.UNICODE)  # remove punctuation (keeps letters/digits/_)
_ws_re = re.compile(r"\s+", flags=re.UNICODE)

def normalize_text(text: str) -> str:
    """
    Normalization for fair ASR evaluation:
    - Unicode normalize (NFKC)
    - lowercase
    - remove punctuation
    - collapse whitespace
    """
    if text is None:
        return ""

    # Normalize Unicode (e.g., combined characters, fancy quotes)
    text = unicodedata.normalize("NFKC", text)

    # Lowercase
    text = text.lower()

    # Remove punctuation (keep letters/digits/underscore/whitespace)
    text = _punct_re.sub(" ", text)

    # Collapse whitespace
    text = _ws_re.sub(" ", text).strip()

    return text


def read_text(path: str) -> str:
    # errors="ignore" prevents crashes from non-UTF8 artifacts
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()


def main():
    ap = argparse.ArgumentParser(
        description="Compute RAW and NORMALIZED WER/CER for hypothesis transcripts against refs."
    )
    ap.add_argument("--lang", choices=["en", "tr"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--hyp_dir", required=True)
    ap.add_argument("--out", default=None, help="Optional output CSV path.")
    args = ap.parse_args()

    base_dir = os.path.dirname(__file__)
    ref_dir = os.path.join(base_dir, "refs", args.lang)

    if not os.path.isdir(ref_dir):
        raise FileNotFoundError(f"Reference directory not found: {ref_dir}")
    if not os.path.isdir(args.hyp_dir):
        raise FileNotFoundError(f"Hypothesis directory not found: {args.hyp_dir}")

    ref_files = sorted(glob.glob(os.path.join(ref_dir, "*.txt")))
    if not ref_files:
        raise FileNotFoundError(f"No .txt reference files found in: {ref_dir}")

    rows = []
    missing = 0

    for ref_path in ref_files:
        fname = os.path.basename(ref_path)
        hyp_path = os.path.join(args.hyp_dir, fname)

        if not os.path.exists(hyp_path):
            print(f"[WARN] Missing hypothesis for: {fname}")
            missing += 1
            continue

        ref_raw = read_text(ref_path)
        hyp_raw = read_text(hyp_path)

        ref_norm = normalize_text(ref_raw)
        hyp_norm = normalize_text(hyp_raw)

        rows.append({
            "model": args.model,
            "lang": args.lang,
            "file": fname,

            # RAW
            "wer_raw": wer(ref_raw, hyp_raw),
            "cer_raw": cer(ref_raw, hyp_raw),

            # NORMALIZED
            "wer_norm": wer(ref_norm, hyp_norm),
            "cer_norm": cer(ref_norm, hyp_norm),

            # lengths (optional, helps diagnose)
            "ref_words_raw": len(ref_raw.split()),
            "hyp_words_raw": len(hyp_raw.split()),
            "ref_words_norm": len(ref_norm.split()),
            "hyp_words_norm": len(hyp_norm.split()),
        })

    df = pd.DataFrame(rows)

    if args.out is None:
        safe_model = args.model.replace("/", "_").replace(" ", "_")
        out_csv = os.path.join(base_dir, f"metrics_{safe_model}_{args.lang}.csv")
    else:
        out_csv = args.out

    df.to_csv(out_csv, index=False)

    print("\n====================")
    print("Saved:", out_csv)
    print(f"Matched files: {len(df)} / {len(ref_files)} (missing: {missing})")
    if len(df):
        print("Avg WER (RAW):", df["wer_raw"].mean())
        print("Avg CER (RAW):", df["cer_raw"].mean())
        print("Avg WER (NORM):", df["wer_norm"].mean())
        print("Avg CER (NORM):", df["cer_norm"].mean())
    print("====================\n")


if __name__ == "__main__":
    main()
