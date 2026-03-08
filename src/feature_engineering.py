import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def feature_engineering(business_df, review_df):
    print("Feature Engineering...")
    
    # 1. Aggregate Reviews per Business
    review_agg = review_df.groupby('business_id').agg({
        'sentiment': ['mean', 'min', 'max', 'std'],
        'stars': ['mean', 'count'] # User avg ratings for the business (proxy for target, be careful of leakage if training)
    }).reset_index()
    
    # Flatten multi-level columns
    review_agg.columns = ['business_id', 'sent_mean', 'sent_min', 'sent_max', 'sent_std', 'avg_user_rating', 'review_count_real']
    
    # 2. Merge with Business Data
    df = pd.merge(business_df, review_agg, on='business_id', how='left')
    
    # Fill NA (for businesses with no reviews, though synthetic data guarantees some)
    df = df.fillna(0)
    
    # 3. Encode Categorical
    le = LabelEncoder()
    df['category_encoded'] = le.fit_transform(df['category'])
    df['city_encoded'] = le.fit_transform(df['city'])
    
    return df
