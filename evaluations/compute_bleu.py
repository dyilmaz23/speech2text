import argparse
import pandas as pd
import sacrebleu

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with columns: text_id, term, reference, candidate, model")
    ap.add_argument("--out", default="evaluations/bleu_summary.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    required = {"text_id", "term", "reference", "candidate", "model"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns: {missing}")

    rows = []
    # compute corpus BLEU per (model, text_id)
    for (model, text_id), g in df.groupby(["model", "text_id"]):
        refs = g["reference"].astype(str).tolist()
        hyps = g["candidate"].astype(str).tolist()

        # sacrebleu expects list of hypotheses and list-of-reference-lists
        bleu = sacrebleu.corpus_bleu(hyps, [refs])
        rows.append({
            "model": model,
            "text_id": text_id,
            "n_terms": len(g),
            "corpus_bleu": bleu.score
        })

    out_df = pd.DataFrame(rows).sort_values(["model", "text_id"])
    out_df.to_csv(args.out, index=False)
    print(f"Saved: {args.out}")
    print(out_df.to_string(index=False))

if __name__ == "__main__":
    main()
