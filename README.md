# Restaurant Rating Prediction

Predicting Yelp star ratings using Business features and Review Sentiment analysis.

## Project Structure
- `data/`: Synthetic Yelp data (Business and Reviews).
- `models/`: ML models.
- `plots/`: Evaluation visualizations.
- `src/`: Code modules.

## Features
- **NLP**: Extracts Sentiment Score (VADER/TextBlob) from reviews.
- **Aggregations**: Computes mean sentiment, review count, etc. per business.
- **Models**: Linear Regression, Random Forest, XGBoost.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
Run the pipeline:
```bash
python main.py
```
