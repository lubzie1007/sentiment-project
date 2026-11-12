"""
Generic Sentiment Analyzer
--------------------------
Analyzes ANY dataset (CSV or TXT) using Hugging Face RoBERTa.
Supports 3-class sentiment: positive, neutral, negative.

Usage:
    python analyze_bert.py --input ../data/reviews.csv
    python analyze_bert.py --input ../data/reviews.txt
    python analyze_bert.py --input ../data/reviews.csv --column "review_body"
    python analyze_bert.py --input ../data/reviews.csv --limit 200

Outputs (saved to ../web/):
    - results.json        Every review + predicted sentiment
    - metrics.json        Accuracy, precision, recall, F1 (only if labels exist)
    - dataset_info.json   Metadata about the uploaded file
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from transformers import pipeline


# ---------- CONFIG ----------
# 3-class model: negative (LABEL_0), neutral (LABEL_1), positive (LABEL_2)
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

DEFAULT_LIMIT = 500
COMMON_TEXT_COLUMNS = [
    "text", "review", "reviews", "comment", "comments",
    "content", "body", "message", "tweet", "post",
    "feedback", "description", "opinion"
]
COMMON_LABEL_COLUMNS = [
    "sentiment", "label", "target", "polarity", "class"
]


# ---------- HELPERS ----------
def clean_text(text):
    """Remove HTML tags and normalize whitespace."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def truncate(text, max_words=300):
    """Models have token limits; truncate very long reviews."""
    words = str(text).split()
    return " ".join(words[:max_words])


def detect_text_column(df, override=None):
    if override:
        if override not in df.columns:
            raise ValueError(f"Column '{override}' not found. Available: {list(df.columns)}")
        return override

    lower_cols = {c.lower(): c for c in df.columns}
    for candidate in COMMON_TEXT_COLUMNS:
        if candidate in lower_cols:
            return lower_cols[candidate]

    print("[!] No obvious text column. Auto-picking the column with longest average text.")
    best_col, best_avg = None, 0
    for col in df.columns:
        try:
            avg_len = df[col].astype(str).str.len().mean()
            if avg_len > best_avg:
                best_avg, best_col = avg_len, col
        except Exception:
            continue
    if not best_col:
        raise ValueError("Could not detect a text column.")
    print(f"[✓] Using column: '{best_col}' (avg length: {best_avg:.1f})")
    return best_col


def detect_label_column(df, text_col, override=None):
    if override:
        if override not in df.columns:
            raise ValueError(f"Column '{override}' not found.")
        return override
    lower_cols = {c.lower(): c for c in df.columns}
    for candidate in COMMON_LABEL_COLUMNS:
        if candidate in lower_cols and lower_cols[candidate] != text_col:
            return lower_cols[candidate]
    return None


def normalize_label(value):
    if pd.isna(value):
        return None
    v = str(value).strip().lower()
    if v in ("positive", "pos", "1", "true", "good"):
        return "positive"
    if v in ("negative", "neg", "0", "false", "bad"):
        return "negative"
    if v in ("neutral", "neu", "2"):
        return "neutral"
    return None


def map_model_label(label):
    """
    Map model output to our standard labels.
    cardiffnlp model returns LABEL_0 (negative), LABEL_1 (neutral), LABEL_2 (positive)
    Newer versions return 'negative', 'neutral', 'positive' directly.
    """
    mapping = {
        'LABEL_0': 'negative',
        'LABEL_1': 'neutral',
        'LABEL_2': 'positive',
        'negative': 'negative',
        'neutral': 'neutral',
        'positive': 'positive',
    }
    return mapping.get(label, label.lower())


def load_data(filepath):
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    ext = path.suffix.lower()

    if ext == ".csv":
        df = pd.read_csv(path)
        print(f"[✓] Loaded CSV: {len(df)} rows, {len(df.columns)} columns")
        print(f"    Columns: {list(df.columns)}")
        return df, "csv"

    elif ext in (".txt", ".text"):
        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        df = pd.DataFrame({"text": lines})
        print(f"[✓] Loaded TXT: {len(df)} lines")
        return df, "txt"

    elif ext in (".xls", ".xlsx"):
        df = pd.read_excel(path)
        print(f"[✓] Loaded Excel: {len(df)} rows, {len(df.columns)} columns")
        return df, "excel"

    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .csv, .txt, or .xlsx")


# ---------- MAIN ----------
def main():
    parser = argparse.ArgumentParser(
        description="Analyze sentiment of any CSV or TXT file using RoBERTa (3-class)."
    )
    parser.add_argument("--input", required=True, help="Path to input file (.csv, .txt, .xlsx)")
    parser.add_argument("--column", default=None, help="Name of the text column")
    parser.add_argument("--label-column", default=None, help="Name of the sentiment label column (optional)")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help=f"Max rows to analyze (default: {DEFAULT_LIMIT})")
    args = parser.parse_args()

    overall_start = time.time()

    # ---------- LOAD ----------
    print("\n=== Loading Dataset ===")
    df, file_type = load_data(args.input)

    if len(df) == 0:
        print("[✗] File is empty.")
        sys.exit(1)

    # ---------- DETECT COLUMNS ----------
    print("\n=== Detecting Columns ===")
    if file_type == "txt":
        text_col = "text"
        label_col = None
    else:
        text_col = detect_text_column(df, override=args.column)
        label_col = detect_label_column(df, text_col, override=args.label_column)

    print(f"[✓] Text column:  '{text_col}'")
    if label_col:
        print(f"[✓] Label column: '{label_col}'")
    else:
        print("[i] No label column found — accuracy metrics will be skipped.")

    # ---------- LIMIT ----------
    original_count = len(df)
    if len(df) > args.limit:
        print(f"\n[i] Limiting to {args.limit} rows (out of {original_count})")
        df = df.sample(n=args.limit, random_state=42).reset_index(drop=True)

    # ---------- CLEAN ----------
    print("\n=== Cleaning Text ===")
    df["_clean_text"] = df[text_col].apply(clean_text).apply(truncate)
    df = df[df["_clean_text"].str.len() > 0].reset_index(drop=True)
    print(f"[✓] {len(df)} rows remain after cleaning")

    # ---------- LOAD MODEL ----------
    print(f"\n=== Loading RoBERTa Model (3-class) ===")
    print(f"    Model: {MODEL_NAME}")
    print(f"    (First run downloads ~500MB, then cached)")
    model_start = time.time()
    classifier = pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        truncation=True,
        max_length=512
    )
    print(f"[✓] Model loaded in {time.time() - model_start:.1f}s")

    # ---------- ANALYZE ----------
    print(f"\n=== Analyzing {len(df)} Reviews ===")
    analyze_start = time.time()
    predictions = classifier(df["_clean_text"].tolist(), batch_size=16)
    analyze_time = time.time() - analyze_start
    print(f"[✓] Analyzed in {analyze_time:.1f}s ({analyze_time/len(df)*1000:.0f}ms per review)")

    # ---------- BUILD RESULTS ----------
    df["_predicted"] = [map_model_label(p["label"]) for p in predictions]
    df["_confidence"] = [p["score"] for p in predictions]

    results_df = pd.DataFrame({
        "text": df["_clean_text"],
        "predicted": df["_predicted"],
        "score": df["_confidence"],
    })

    if label_col:
        results_df["actual"] = df[label_col].apply(normalize_label)

    # ---------- SAVE ----------
    print("\n=== Saving Output ===")
    out_dir = Path("../web")
    out_dir.mkdir(exist_ok=True)

    results_records = results_df.to_dict(orient="records")
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results_records, f, indent=2)
    print(f"[✓] Saved {len(results_records)} rows to web/results.json")

    # ---------- METRICS (if labels exist) ----------
    has_labels = "actual" in results_df.columns and results_df["actual"].notna().any()

    if has_labels:
        eval_df = results_df.dropna(subset=["actual"]).copy()
        # Keep only rows where both actual and predicted are in our 3 classes
        eval_df = eval_df[eval_df["actual"].isin(["positive", "negative", "neutral"])]
        eval_df = eval_df[eval_df["predicted"].isin(["positive", "negative", "neutral"])]

        # Confusion matrix (3x3 if neutral present, else 2x2)
        classes = sorted(eval_df["actual"].unique().tolist())
        cm = {}
        for a in classes:
            for p in classes:
                cm[f"{a}_{p}"] = int(((eval_df["actual"] == a) & (eval_df["predicted"] == p)).sum())

        # Accuracy
        total = len(eval_df)
        correct = int((eval_df["actual"] == eval_df["predicted"]).sum())
        accuracy = correct / total if total else 0

        # Per-class precision/recall/F1 (macro average for simplicity)
        metrics_per_class = {}
        precisions, recalls, f1s = [], [], []
        for cls in classes:
            tp = int(((eval_df["actual"] == cls) & (eval_df["predicted"] == cls)).sum())
            fp = int(((eval_df["actual"] != cls) & (eval_df["predicted"] == cls)).sum())
            fn = int(((eval_df["actual"] == cls) & (eval_df["predicted"] != cls)).sum())
            p = tp / (tp + fp) if (tp + fp) else 0
            r = tp / (tp + fn) if (tp + fn) else 0
            f1 = 2 * (p * r) / (p + r) if (p + r) else 0
            metrics_per_class[cls] = {
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1_score": round(f1, 4)
            }
            precisions.append(p); recalls.append(r); f1s.append(f1)

        macro_p = sum(precisions) / len(precisions) if precisions else 0
        macro_r = sum(recalls) / len(recalls) if recalls else 0
        macro_f1 = sum(f1s) / len(f1s) if f1s else 0

        metrics = {
            "model": MODEL_NAME,
            "sample_size": int(total),
            "classes": classes,
            "confusion_matrix": cm,
            "metrics": {
                "accuracy": round(accuracy, 4),
                "precision": round(macro_p, 4),
                "recall": round(macro_r, 4),
                "f1_score": round(macro_f1, 4),
                "total_evaluated": int(total),
            },
            "per_class": metrics_per_class,
        }

        with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        print(f"[✓] Metrics saved to web/metrics.json")
        print(f"    Accuracy:  {accuracy:.2%}")
        print(f"    Precision: {macro_p:.2%} (macro)")
        print(f"    Recall:    {macro_r:.2%} (macro)")
        print(f"    F1 Score:  {macro_f1:.2%} (macro)")
    else:
        with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump({"available": False, "reason": "No label column found in dataset"}, f, indent=2)
        print("[i] No labels found — skipping metrics")

    # ---------- DATASET INFO ----------
    total_time = time.time() - overall_start
    dataset_info = {
        "source_file": os.path.basename(args.input),
        "file_type": file_type,
        "analyzed_at": datetime.now().isoformat(timespec="seconds"),
        "total_rows_in_file": int(original_count),
        "rows_analyzed": int(len(results_df)),
        "text_column": str(text_col),
        "has_labels": bool(has_labels),
        "label_column": str(label_col) if has_labels else None,
        "model": MODEL_NAME,
        "classes": ["positive", "neutral", "negative"],
        "processing_time_seconds": round(float(total_time), 1),
    }

    with open(out_dir / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2)
    print(f"[✓] Dataset info saved to web/dataset_info.json")

    # ---------- SUMMARY ----------
    print("\n" + "=" * 50)
    print("DONE")
    print("=" * 50)
    print(f"File analyzed:  {dataset_info['source_file']}")
    print(f"Rows analyzed:  {len(results_df)}")
    print(f"Total time:     {total_time:.1f}s")
    print(f"\nOpen your dashboard to see the results.")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()