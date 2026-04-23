# Beirut Dining Analysis

An end-to-end data science project that scrapes, cleans, analyzes, and visualizes restaurant data from across Lebanon.

🔗 **Live Dashboard:** [dining-analysis-26.streamlit.app](https://dining-analysis-26.streamlit.app/)

---

## Overview

This project aggregates data from three platforms — **Wandorlog**, **Restaurant Guru**, and **TripAdvisor** — into a unified dataset of restaurants and reviews, then applies NLP and machine learning to extract insights.

---

## Pipeline

```
Scraping → Cleaning → Merging → Geocoding → Sentiment Analysis → ML → Dashboard
```

## Tech Stack

`Selenium` · `BeautifulSoup` · `Pandas` · `NumPy` · `scikit-learn` · `LightGBM` · `TextBlob` · `spaCy` · `NLTK` · `Matplotlib` · `Seaborn` · `Plotly` · `Streamlit`

---

## Project Structure

```
├── scrapers/          # Selenium scrapers (Wandorlog, Guru, TripAdvisor)
├── Cleaners/          # Per-source cleaning scripts
├── merged/            # Merge + deduplication
├── sentiment/         # Sentiment analysis
├── machine_learning/  # Cuisine classifier
├── nlp/               # Keyword extraction
├── dashboard/         # Streamlit app + geocoding
├── cleaned/           # Cleaned CSVs (auto-generated)
└── .github/workflows/ # GitHub Actions pipeline
```

---

## Course

COSC 482 – Data Science and Web Scraping | Rafik Hariri University | Spring 2026