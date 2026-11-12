"""
Flask server for the Sentiment Analyzer.
Serves the website AND handles file uploads + analysis.
Now with Gemini-powered recommendations.

Run: python server.py
Then visit: http://localhost:5000
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from transformers import pipeline
import pandas as pd
from pathlib import Path
from collections import Counter
import os, re, json, time
from datetime import datetime

# Gemini imports
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("[!] google-genai not installed. Recommendations will be disabled.")
    print("    Install with: pip install google-genai")


# ---------- CONFIG ----------
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
GEMINI_MODEL = "gemini-3.5-flash-lite"

WEB_DIR    = Path(__file__).parent.parent / "web"
UPLOAD_DIR = Path(__file__).parent.parent / "uploads"

COMMON_TEXT_COLUMNS = ["text", "review", "reviews", "comment", "comments",
                       "content", "body", "message", "tweet", "post",
                       "feedback", "description", "opinion"]
COMMON_LABEL_COLUMNS = ["sentiment", "label", "target", "polarity", "class"]

# Stop words for keyword extraction
STOPWORDS = set([
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "her", "was",
    "one", "our", "out", "day", "get", "has", "him", "his", "how", "man", "new",
    "now", "old", "see", "two", "way", "who", "boy", "did", "its", "let", "put",
    "say", "she", "too", "use", "that", "this", "with", "have", "from", "they",
    "will", "would", "there", "their", "what", "about", "which", "when", "your",
    "them", "than", "then", "into", "more", "some", "could", "other", "been",
    "were", "also", "just", "only", "very", "much", "even", "like", "well",
    "make", "made", "back", "still", "over", "after", "before", "because",
    "these", "those", "being", "every", "does", "here", "most", "such",
    "movie", "film", "watch", "watching", "watched", "really", "quite",
    "thing", "things", "people", "going", "even", "though", "still"
])

UPLOAD_DIR.mkdir(exist_ok=True)


# ---------- FLASK APP ----------
app = Flask(__name__, static_folder=str(WEB_DIR), static_url_path="")
CORS(app)

# Load sentiment model ONCE at startup
print("=" * 60)
print("Loading RoBERTa sentiment model...")
print("(First run downloads ~500MB, then cached)")
print("=" * 60)
model_start = time.time()
classifier = pipeline(
    "sentiment-analysis",
    model=MODEL_NAME,
    truncation=True,
    max_length=512
)
print(f"✓ Sentiment model ready in {time.time() - model_start:.1f}s")

# ---------- INITIALIZE GEMINI ----------
# Tries: config.py first, then environment variable
gemini_client = None
if GEMINI_AVAILABLE:
    api_key = None
    key_source = None

    # Try config.py first
    try:
        from config import GEMINI_API_KEY as api_key
        key_source = "config.py"
    except ImportError:
        # Fall back to environment variable
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            key_source = "environment variable"

    if api_key and api_key != "PASTE-YOUR-ACTUAL-KEY-HERE":
        try:
            gemini_client = genai.Client(api_key=api_key)
            print(f"✓ Gemini client initialized (key from {key_source})")
        except Exception as e:
            print(f"[!] Gemini client failed: {e}")
            gemini_client = None
    else:
        print("[!] No Gemini API key found. Recommendations will be disabled.")
        print("    Create python/config.py with: GEMINI_API_KEY = 'your-key'")
        print("    Or set the GEMINI_API_KEY environment variable.")
        print("    Get a key at: https://aistudio.google.com/apikey")

print("=" * 60)


# ---------- HELPERS ----------
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def truncate(text, max_words=300):
    return " ".join(str(text).split()[:max_words])


def map_model_label(label):
    mapping = {
        'LABEL_0': 'negative', 'LABEL_1': 'neutral', 'LABEL_2': 'positive',
        'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive',
    }
    return mapping.get(label, label.lower())


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


def detect_text_column(df, override=None):
    if override and override in df.columns:
        return override
    lower_cols = {c.lower(): c for c in df.columns}
    for candidate in COMMON_TEXT_COLUMNS:
        if candidate in lower_cols:
            return lower_cols[candidate]
    best_col, best_avg = None, 0
    for col in df.columns:
        try:
            avg_len = df[col].astype(str).str.len().mean()
            if avg_len > best_avg:
                best_avg, best_col = avg_len, col
        except Exception:
            continue
    return best_col or df.columns[0]


def detect_label_column(df, text_col, override=None):
    if override and override in df.columns:
        return override
    lower_cols = {c.lower(): c for c in df.columns}
    for candidate in COMMON_LABEL_COLUMNS:
        if candidate in lower_cols and lower_cols[candidate] != text_col:
            return lower_cols[candidate]
    return None


def get_top_words(text_series, top_n=8):
    """Extract top N meaningful words from a series of texts."""
    all_words = []
    for text in text_series:
        words = re.findall(r'\b[a-z]{3,}\b', str(text).lower())
        all_words.extend([w for w in words if w not in STOPWORDS])
    return [word for word, _ in Counter(all_words).most_common(top_n)]


def generate_recommendations(stats):
    """Call Gemini to generate recommendations based on dataset stats."""
    if not gemini_client:
        return None

    prompt = f"""You are a senior business analyst reviewing customer feedback.

## Data Summary
- Total reviews analyzed: {stats['total']}
- Sentiment breakdown: {stats['pos_pct']}% positive, {stats['neu_pct']}% neutral, {stats['neg_pct']}% negative
- Average model confidence: {stats['avg_conf']}%
- Top positive keywords: {', '.join(stats['pos_words']) if stats['pos_words'] else 'none'}
- Top negative keywords: {', '.join(stats['neg_words']) if stats['neg_words'] else 'none'}
- Review category: {stats['category']}

## Your Task
Write exactly 3 actionable recommendations. Each should:
1. Be specific (mention actual themes from the data)
2. Be prioritized (most important first)
3. Be concise (one sentence each)

Format your response EXACTLY like this (no other text):
1. [RECOMMENDATION] — [1-line justification]
2. [RECOMMENDATION] — [1-line justification]
3. [RECOMMENDATION] — [1-line justification]
"""

    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        return response.text.strip()
    except Exception as e:
        print(f"[!] Gemini API error: {e}")
        return None


# ---------- ROUTES: STATIC FILES ----------
@app.route("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(WEB_DIR, filename)


# ---------- ROUTE: GET current data ----------
@app.route("/api/info")
def get_info():
    info_path = WEB_DIR / "dataset_info.json"
    if not info_path.exists():
        return jsonify({"available": False})
    with open(info_path) as f:
        return jsonify(json.load(f))


# ---------- ROUTE: ANALYZE UPLOADED FILE ----------
@app.route("/api/analyze", methods=["POST"])
def analyze_file():
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"}), 400

        file = request.files["file"]
        limit = int(request.form.get("limit", 500))
        text_col_override = request.form.get("column") or None
        category = request.form.get("category") or "general"

        if file.filename == "":
            return jsonify({"success": False, "error": "Empty filename"}), 400

        # Save uploaded file
        safe_name = re.sub(r'[^a-zA-Z0-9._-]', '_', file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        upload_path = UPLOAD_DIR / f"{timestamp}_{safe_name}"
        file.save(upload_path)

        # Load data
        ext = Path(safe_name).suffix.lower()
        if ext == ".csv":
            df = pd.read_csv(upload_path)
            file_type = "csv"
        elif ext in (".txt", ".text"):
            with open(upload_path, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
            df = pd.DataFrame({"text": lines})
            file_type = "txt"
        elif ext in (".xls", ".xlsx"):
            df = pd.read_excel(upload_path)
            file_type = "excel"
        else:
            return jsonify({"success": False, "error": f"Unsupported file type: {ext}"}), 400

        if len(df) == 0:
            return jsonify({"success": False, "error": "File is empty"}), 400

        original_count = len(df)

        # Detect columns
        if file_type == "txt":
            text_col = "text"
            label_col = None
        else:
            text_col = detect_text_column(df, text_col_override)
            label_col = detect_label_column(df, text_col)

        # Limit rows
        if len(df) > limit:
            df = df.sample(n=limit, random_state=42).reset_index(drop=True)

        # Clean text
        df["_clean_text"] = df[text_col].apply(clean_text).apply(truncate)
        df = df[df["_clean_text"].str.len() > 0].reset_index(drop=True)

        # Run BERT
        analyze_start = time.time()
        predictions = classifier(df["_clean_text"].tolist(), batch_size=16)
        analyze_time = time.time() - analyze_start

        df["_predicted"] = [map_model_label(p["label"]) for p in predictions]
        df["_confidence"] = [p["score"] for p in predictions]

        results_df = pd.DataFrame({
            "text": df["_clean_text"],
            "predicted": df["_predicted"],
            "score": df["_confidence"],
            "category": [category] * len(df),
        })

        if label_col:
            results_df["actual"] = df[label_col].apply(normalize_label)

        # Save results.json
        results_records = results_df.to_dict(orient="records")
        with open(WEB_DIR / "results.json", "w", encoding="utf-8") as f:
            json.dump(results_records, f, indent=2)

        # Metrics if labels exist
        has_labels = "actual" in results_df.columns and results_df["actual"].notna().any()

        if has_labels:
            eval_df = results_df.dropna(subset=["actual"]).copy()
            eval_df = eval_df[eval_df["actual"].isin(["positive", "negative", "neutral"])]
            eval_df = eval_df[eval_df["predicted"].isin(["positive", "negative", "neutral"])]

            classes = sorted(eval_df["actual"].unique().tolist())
            cm = {}
            for a in classes:
                for p in classes:
                    cm[f"{a}_{p}"] = int(((eval_df["actual"] == a) & (eval_df["predicted"] == p)).sum())

            total = len(eval_df)
            correct = int((eval_df["actual"] == eval_df["predicted"]).sum())
            accuracy = correct / total if total else 0

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

            macro_p  = sum(precisions) / len(precisions) if precisions else 0
            macro_r  = sum(recalls) / len(recalls) if recalls else 0
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
            with open(WEB_DIR / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
        else:
            with open(WEB_DIR / "metrics.json", "w", encoding="utf-8") as f:
                json.dump({"available": False, "reason": "No label column found"}, f, indent=2)

        # Save dataset_info.json
        dataset_info = {
            "source_file": str(file.filename),
            "file_type": str(file_type),
            "analyzed_at": datetime.now().isoformat(timespec="seconds"),
            "total_rows_in_file": int(original_count),
            "rows_analyzed": int(len(results_df)),
            "text_column": str(text_col),
            "has_labels": bool(has_labels),
            "label_column": str(label_col) if has_labels else None,
            "model": str(MODEL_NAME),
            "category": str(category),
            "classes": ["positive", "neutral", "negative"],
            "processing_time_seconds": round(float(analyze_time), 1),
        }
        with open(WEB_DIR / "dataset_info.json", "w", encoding="utf-8") as f:
            json.dump(dataset_info, f, indent=2)

        # ---------- GENERATE RECOMMENDATIONS ----------
        pos_count = int((results_df["predicted"] == "positive").sum())
        neg_count = int((results_df["predicted"] == "negative").sum())
        neu_count = int((results_df["predicted"] == "neutral").sum())
        total_count = len(results_df)

        top_pos_words = get_top_words(results_df[results_df["predicted"] == "positive"]["text"])
        top_neg_words = get_top_words(results_df[results_df["predicted"] == "negative"]["text"])

        stats = {
            "total": total_count,
            "pos_pct": round((pos_count / total_count) * 100, 1) if total_count else 0,
            "neu_pct": round((neu_count / total_count) * 100, 1) if total_count else 0,
            "neg_pct": round((neg_count / total_count) * 100, 1) if total_count else 0,
            "avg_conf": round(float(results_df["score"].mean()) * 100, 1) if total_count else 0,
            "pos_words": top_pos_words,
            "neg_words": top_neg_words,
            "category": category,
        }

        print(f"\n[i] Generating recommendations from Gemini...")
        recommendations_text = generate_recommendations(stats)

        if recommendations_text:
            with open(WEB_DIR / "recommendations.json", "w", encoding="utf-8") as f:
                json.dump({
                    "text": recommendations_text,
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                    "model": GEMINI_MODEL,
                }, f, indent=2)
            print(f"[✓] Recommendations saved")
        else:
            with open(WEB_DIR / "recommendations.json", "w", encoding="utf-8") as f:
                json.dump({
                    "text": "Recommendations unavailable — Gemini API key not set or error occurred.",
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                    "model": None,
                }, f, indent=2)
            print(f"[i] Recommendations skipped (no API key or error)")

        # Clean up upload
        try:
            upload_path.unlink()
        except Exception:
            pass

        return jsonify({
            "success": True,
            "file_name": str(file.filename),
            "rows_analyzed": int(len(results_df)),
            "has_labels": bool(has_labels),
            "category": str(category),
            "processing_time": round(float(analyze_time), 1),
            "positive": pos_count,
            "neutral": neu_count,
            "negative": neg_count,
            "has_recommendations": bool(recommendations_text),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ---------- ROUTE: ANALYZE SINGLE TEXT ----------
@app.route("/api/analyze-text", methods=["POST"])
def analyze_text():
    data = request.get_json()
    text = data.get("text", "").strip()
    category = data.get("category", "general")
    if not text:
        return jsonify({"success": False, "error": "No text provided"}), 400

    result = classifier(text, truncation=True)[0]
    return jsonify({
        "success": True,
        "label": map_model_label(result["label"]),
        "score": float(result["score"]),
        "category": str(category),
    })


# ---------- MAIN ----------
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Sentiment Analyzer — Flask Server")
    print("=" * 60)
    print(f"  Serving:  {WEB_DIR}")
    print(f"  Model:    {MODEL_NAME}")
    print(f"  Gemini:   {'✓ ready' if gemini_client else '✗ not configured'}")
    print(f"  URL:      http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=False)