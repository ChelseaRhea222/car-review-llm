import os
import re
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

# ====== 1) READ RAW BYTES + DECODE SAFELY ======
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

# ====== 3) SUMMARIZE REVIEWS (SUPPORTED TASK: text-generation) ======
# NOTE: This is not as strong as a true summarization model, but it WILL run
# because your Transformers install supports 'text-generation'.
generator = pipeline("text-generation", model="distilgpt2")

def summarize_one(review_text: str) -> str:
    prompt = (
        "Write a single-sentence summary of this car review. "
        "Do not add extra commentary.\n"
        f"Review: {review_text}\n"
        "Summary:"
    )

    out = generator(
        prompt,
        max_new_tokens=35,
        do_sample=False,
        num_return_sequences=1,
        pad_token_id=50256
    )[0]["generated_text"]

    # Grab text after "Summary:"
    summary = out.split("Summary:", 1)[-1].strip()

    # Keep it to one sentence (basic cleanup)
    if "." in summary:
        summary = summary.split(".", 1)[0].strip() + "."

    # Fallback if model returns nothing useful
    if len(summary) < 5:
        summary = "Summary unavailable."

    return summary

df["Summary"] = [summarize_one(r) for r in df["Review"].tolist()]

# ====== 4) SAVE OUTPUT FILE YOU CAN UPLOAD ======
df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
print("Saved summarized output file:", OUTPUT_PATH)
