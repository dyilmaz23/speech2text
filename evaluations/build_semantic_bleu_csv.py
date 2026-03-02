import argparse
import json
import csv
import re
from pathlib import Path

def normalize_term(t: str) -> str:
    t = t.strip()
    # remove markdown bold like **Term**
    t = t.replace("**", "").strip()
    # remove parenthetical abbreviations: "Large Language Models (LLMs)" -> "Large Language Models"
    t = re.sub(r"\s*\([^)]*\)\s*$", "", t).strip()
    # unify whitespace
    t = re.sub(r"\s+", " ", t)
    return t

def parse_terms_numbered_list(text: str) -> dict:
    """
    Parses lines like:
    1. **Term**: explanation...
    2. Term: explanation...
    """
    terms = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^\d+\.\s*(.+?)\s*:\s*(.+)$", line)
        if not m:
            continue
        raw_term = m.group(1).strip()
        expl = m.group(2).strip()
        term = normalize_term(raw_term)
        if term and expl:
            terms[term] = expl
    return terms

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--semantic_root", required=True, help="semantic-exp/outputs/en")
    ap.add_argument("--oxford_refs", required=True, help="evaluations/oxford_refs_en.json")
    ap.add_argument("--out", default="evaluations/semantic_bleu_en.csv")
    args = ap.parse_args()

    semantic_root = Path(args.semantic_root)
    refs = json.loads(Path(args.oxford_refs).read_text(encoding="utf-8"))

    rows = []
    skipped_no_ref = 0
    total_terms_seen = 0

    for model_dir in sorted([p for p in semantic_root.iterdir() if p.is_dir()]):
        model = model_dir.name
        for txt in sorted(model_dir.glob("*.txt")):
            text_id = txt.stem
            content = txt.read_text(encoding="utf-8", errors="replace")
            terms = parse_terms_numbered_list(content)

            for term, candidate in terms.items():
                total_terms_seen += 1
                ref = refs.get(term)
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
