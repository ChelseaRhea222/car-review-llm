# LLM Car Review Summaries

A lightweight NLP pipeline that cleans messy customer review CSV files and uses a small language model to automatically generate concise one-sentence summaries for each review.

This project demonstrates practical **data cleaning, fault-tolerant parsing, and applied NLP** to transform raw text into analysis-ready insights.

---

## What this does

Input:

* Raw CSV with long customer reviews
* Inconsistent formatting and delimiters

Pipeline:

1. Detect file encoding automatically
2. Clean + normalize text
3. Parse review + sentiment label safely
4. Generate 1-sentence summary using HuggingFace transformers
5. Export clean dataset

Output:

* `car_review_summaries.csv`
* Ready for dashboards or analysis

---

## Tech Stack

* Python
* Pandas
* HuggingFace Transformers
* CSV / Text Processing
* NLP

---

## Project Structure

```
car-review-llm/
│
├── data/
│   └── car_reviews.csv
│
├── car_review_summaries.csv
├── llm_car_review.py
└── README.md
```

---

##  Run locally

Install dependencies:

```
pip install pandas transformers
```

Run:

```
python llm_car_review.py
```

---

## Skills Demonstrated

* Data wrangling
* Robust CSV parsing
* Text preprocessing
* Transformer-based NLP
* End-to-end data pipeline design

---

## Author

Chelsea Rhea
Data Analytics & Engineering Portfolio

