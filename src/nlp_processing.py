from textblob import TextBlob
import pandas as pd
import re

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def get_sentiment(text):
    """
    Returns polarity score between -1 and 1.
    """
    return TextBlob(text).sentiment.polarity

def process_reviews(review_df):
    print("Processing reviews (cleaning + sentiment)...")
    review_df['clean_text'] = review_df['text'].apply(clean_text)
    review_df['sentiment'] = review_df['clean_text'].apply(get_sentiment)
    return review_df
