import pandas as pd
import numpy as np
import random
import os

def load_data(business_path="data/business.csv", review_path="data/reviews.csv"):
    if not os.path.exists(business_path) or not os.path.exists(review_path):
        print("Generating synthetic Yelp data...")
        generate_synthetic_data(business_path, review_path)
    
    business_df = pd.read_csv(business_path)
    review_df = pd.read_csv(review_path)
    return business_df, review_df

def generate_synthetic_data(b_path, r_path):
    np.random.seed(42)
    random.seed(42)
    
    n_businesses = 200
    n_reviews = 5000
    
    # 1. Generate Business Data
    categories_list = ['Italian', 'Mexican', 'Chinese', 'American', 'Indian', 'Thai', 'Burger', 'Pizza']
    cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
    
    businesses = []
    for bid in range(100, 100 + n_businesses):
        businesses.append({
            'business_id': f"B{bid}",
            'name': f"Restaurant {bid}",
            'category': random.choice(categories_list),
            'city': random.choice(cities),
            'price_range': random.randint(1, 4), # 1 to 4 '$' s
            'latitude': np.random.uniform(30, 45),
            'longitude': np.random.uniform(-120, -70),
            'average_stars': round(np.random.uniform(1.5, 5.0), 1), # True label (noisy)
            'review_count': 0 # Will update later
        })
    b_df = pd.DataFrame(businesses)
    
    # 2. Generate Reviews
    # Reviews should correlate somewhat with the actual stars of the business
    reviews = []
    texts_positive = ["Great food!", "Amazing service.", "Loved it.", "Will come again.", "Delicious."]
    texts_negative = ["Terrible experience.", "Rude staff.", "Cold food.", "Waste of money.", "Never again."]
    texts_neutral = ["It was okay.", "Average food.", "Not bad but not great.", "Decent price.", "Slightly delayed."]
    
    for rid in range(1000, 1000 + n_reviews):
        bus = b_df.sample(1).iloc[0]
        
        # Simulate user rating based on true business rating + noise
        rating = round(np.random.normal(bus['average_stars'], 0.8))
        rating = max(1, min(5, int(rating)))
        
        # Generate text based on rating
        if rating >= 4:
            text = random.choice(texts_positive) + " " + random.choice(texts_positive)
        elif rating <= 2:
            text = random.choice(texts_negative) + " " + random.choice(texts_negative)
        else:
            text = random.choice(texts_neutral)
            
        reviews.append({
            'review_id': f"R{rid}",
            'business_id': bus['business_id'],
            'user_id': f"U{random.randint(1, 500)}",
            'stars': rating,
            'text': text,
            'date': f"202{random.randint(0,5)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
        })
        
    r_df = pd.DataFrame(reviews)
    
    # Update review counts in business df
    counts = r_df['business_id'].value_counts()
    b_df['review_count'] = b_df['business_id'].map(counts).fillna(0).astype(int)
    
    # Save
    os.makedirs(os.path.dirname(b_path), exist_ok=True)
    b_df.to_csv(b_path, index=False)
    r_df.to_csv(r_path, index=False)
    print("Data generated.")
