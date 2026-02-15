import os
import re
from collections import Counter
import pandas as pd
from transformers import pipeline

"""
llm_car_review.py

What this script does:
1) Robustly reads car_reviews.csv even if Excel saved it with odd encodings (utf-8/utf-16/cp1252/etc.)
2) Cleans/parses it into exactly two columns: Review, Class (POSITIVE/NEGATIVE)
   - Handles extra delimiters inside the Review by splitting on the LAST delimiter
3) Summarizes each review using a supported Transformers task in your environment: text-generation
4) Saves ONE deliverable file you can upload: car_review_summaries.csv
"""

# ====== PATHS (matches your folder structure) ======
BASE_DIR = os.path.dirname(__file__)
INPUT_PATH = os.path.join(BASE_DIR, "data", "car_reviews.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "car_review_summaries.csv")

print("Reading file:", INPUT_PATH)
if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(f"Input file not found at: {INPUT_PATH}")

size = os.path.getsize(INPUT_PATH)
print("File size (bytes):", size)
if size == 0:
    raise ValueError("car_reviews.csv is 0 bytes (empty). Re-export or re-download the file.")

# ====== 1) READ RAW BYTES + DECODE ======
raw = open(INPUT_PATH, "rb").read()
print("First 80 bytes:", raw[:80])

text = None
enc_used = None

for enc in ("utf-8-sig", "utf-16", "utf-16le", "utf-16be", "cp1252", "latin1"):
    try:
        candidate = raw.decode(enc)
        if re.sub(r"\s+", "", candidate):
            text = candidate
            enc_used = enc
            break
    except Exception:
        continue

if text is None:
    raise ValueError(
        "Could not decode car_reviews.csv into readable text. "
        "Try re-saving as 'CSV UTF-8 (Comma delimited)' from Excel."
    )

print("Decoded with:", enc_used)

lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
if not lines:
    raise ValueError("After decoding, the file still appears empty.")

first = lines[0].lower()
if ("review" in first and "class" in first) or first.startswith("review") or first.startswith('"review"'):
    lines = lines[1:]

# ====== 2) PARSE INTO Review + Class ======
rows = []
for line in lines:
    # Prefer semicolon format: review ; label
    review, sep, label = line.rpartition(";")

    # If no semicolon, try comma format: review , label
    if sep == "":
        review, sep, label = line.rpartition(",")

    if sep == "":
        continue

    review = review.strip().strip('"')
    label = label.strip().strip('"').upper()

    if label in {"POSITIVE", "NEGATIVE"} and review:
        rows.append({"Review": review, "Class": label})

df = pd.DataFrame(rows)

print("Rows parsed:", len(df))
print("Columns:", df.columns.tolist())
print(df.head())

if df.empty:
    raise ValueError(
        "Parsed 0 rows. Expected format per line:\n"
        "  <review text><delimiter><POSITIVE/NEGATIVE>"
    )

# ====== 3) SUMMARIZE REVIEWS ======
# Prefer a true summarization model; if unavailable, use an extractive fallback.
summarizer = None
try:
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    print("Summarizer: sshleifer/distilbart-cnn-12-6")
except Exception as e:
    print("Summarization model unavailable, using extractive fallback.")
    print("Reason:", str(e))

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for",
    "from", "had", "has", "have", "he", "her", "him", "his", "i", "if", "in",
    "is", "it", "its", "me", "my", "of", "on", "or", "our", "she", "so",
    "that", "the", "their", "them", "there", "they", "this", "to", "was",
    "we", "were", "with", "you", "your"
}

def split_sentences(text: str):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p and re.search(r"[A-Za-z0-9]", p)]

def tokenize(text: str):
    return [w for w in re.findall(r"[A-Za-z']+", text.lower()) if w not in STOPWORDS]

def normalize_sentence(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    if s and s[-1] not in ".!?":
        s += "."
    return s

def extractive_summary(review_text: str) -> str:
    sentences = split_sentences(review_text)
    if not sentences:
        return "Summary unavailable."
    if len(sentences) == 1:
        return normalize_sentence(sentences[0])

    doc_tokens = tokenize(review_text)
    freqs = Counter(doc_tokens)

    best_score = float("-inf")
    best_sentence = sentences[0]

    for idx, sent in enumerate(sentences):
        tokens = tokenize(sent)
        if not tokens:
            continue

        keyword_score = sum(freqs[t] for t in tokens) / max(len(tokens), 1)
        length = len(sent.split())
        # Favor moderately detailed sentences over very short/generic ones.
        length_bonus = 0.0
        if 10 <= length <= 28:
            length_bonus = 1.5
        elif 7 <= length < 10 or 29 <= length <= 35:
            length_bonus = 0.7
        elif length < 5:
            length_bonus = -1.5

        # Small penalty for first sentence to reduce "just copy first line".
        position_penalty = -0.35 if idx == 0 else 0.0
        score = keyword_score + length_bonus + position_penalty

        if score > best_score:
            best_score = score
            best_sentence = sent

    return normalize_sentence(best_sentence)

def summarize_one(review_text: str) -> str:
    review_text = re.sub(r"\s+", " ", str(review_text)).strip()
    if not review_text:
        return "Summary unavailable."

    if summarizer is not None:
        try:
            out = summarizer(
                review_text,
                max_length=48,
                min_length=16,
                do_sample=False,
                truncation=True
            )[0]["summary_text"]
            out = normalize_sentence(out)
            # Guardrail for obvious low-quality generations.
            if len(out.split()) >= 5 and "i am a very good car" not in out.lower():
                return out
        except Exception:
            pass

    return extractive_summary(review_text)

df["Summary"] = [summarize_one(r) for r in df["Review"].tolist()]

# ====== 4) SAVE OUTPUT FILE YOU CAN UPLOAD ======
df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
print("Saved summarized output file:", OUTPUT_PATH)
