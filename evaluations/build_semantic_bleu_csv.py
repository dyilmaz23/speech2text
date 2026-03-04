import argparse
import json
import csv
import re
import unicodedata
from pathlib import Path


def normalize_term(t: str) -> str:
    if t is None:
        return ""

    # unicode normalize (fancy quotes, etc.)
    t = unicodedata.normalize("NFKC", t)

    t = t.strip()

    # remove markdown bold like **Term**
    t = t.replace("**", "").strip()

    # normalize apostrophes: planck’s -> planck's
    t = t.replace("’", "'").replace("`", "'")

    # normalize dashes: stefan–boltzmann / stefan—boltzmann -> stefan-boltzmann
    t = t.replace("–", "-").replace("—", "-")

    # remove parenthetical abbreviations at end:
    # "Large Language Models (LLMs)" -> "Large Language Models"
    t = re.sub(r"\s*\([^)]*\)\s*$", "", t).strip()

    # strip trailing punctuation (common in LLM outputs)
    t = re.sub(r"^[\s\.\,\:\;\-\–\—]+|[\s\.\,\:\;\!\?\-\–\—]+$", "", t)

    # lowercase (your main request)
    t = t.lower()

    # unify whitespace
    t = re.sub(r"\s+", " ", t).strip()

    return t

def parse_terms_flexible(text: str) -> dict:
    """
    Robust parser for common LLM formats. Supports:
      - 1. Term: explanation
      - 1) Term: explanation
      - - Term: explanation
      - - **Term**: explanation
    Also supports multi-line explanations (until next term header).
    """
    terms = {}
    current_term = None
    current_lines = []

    # term header patterns
    header_re = re.compile(
        r"^(?:\s*(?:\d+[\.\)]|\-|\*)\s*)?(.+?)\s*[:\-–]\s*(.*)$"
    )

    def flush():
        nonlocal current_term, current_lines
        if current_term:
            expl = " ".join([ln.strip() for ln in current_lines if ln.strip()]).strip()
            if expl:
                terms[current_term] = expl
        current_term = None
        current_lines = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        m = header_re.match(line)
        if m:
            # start new term
            flush()
            raw_term = m.group(1).strip()
            first_expl = m.group(2).strip()

            term = normalize_term(raw_term)
            if term:
                current_term = term
                if first_expl:
                    current_lines.append(first_expl)
            continue

        # continuation line for current explanation
        if current_term:
            current_lines.append(line)

    flush()
    return terms

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--semantic_root", required=True, help="semantic-exp/outputs/en")
    ap.add_argument("--oxford_refs", required=True, help="evaluations/oxford_refs_en.json")
    ap.add_argument("--out", default="evaluations/semantic_bleu_en.csv")
    args = ap.parse_args()

    semantic_root = Path(args.semantic_root)
    refs_raw = json.loads(Path(args.oxford_refs).read_text(encoding="utf-8"))
    refs = {normalize_term(k): v for k, v in refs_raw.items()}
    # normalize reference keys once for robust matching
    refs_norm = {normalize_term(k): v for k, v in refs.items()}

    rows = []
    skipped_no_ref = 0
    total_terms_seen = 0

    for model_dir in sorted([p for p in semantic_root.iterdir() if p.is_dir()]):
        model = model_dir.name
        for txt in sorted(model_dir.glob("*.txt")):
            text_id = txt.stem
            content = txt.read_text(encoding="utf-8", errors="replace")
            terms = parse_terms_flexible(content)   

            for term, candidate in terms.items():
                total_terms_seen += 1
                ref = refs_norm.get(term)
                if not ref:
                    skipped_no_ref += 1
                    continue
                rows.append({
                    "text_id": text_id,
                    "term": term,
                    "reference": ref,
                    "candidate": candidate,
                    "model": model
                })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["text_id", "term", "reference", "candidate", "model"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {args.out}")
    print(f"Total terms seen: {total_terms_seen}")
    print(f"Rows written (with Oxford refs): {len(rows)}")
    print(f"Skipped (no Oxford ref): {skipped_no_ref}")

if __name__ == "__main__":
    main()
