import pandas as pd
from src.data_loader import load_data
from src.nlp_processing import process_reviews
from src.feature_engineering import feature_engineering
from src.model import train_and_evaluate
from src.visualization import plot_results

def main():
    print("Starting Restaurant Rating Prediction Pipeline...")
    
    # 1. Load Data
    business_df, review_df = load_data()
    print(f"Loaded {len(business_df)} businesses and {len(review_df)} reviews.")
    
    # 2. NLP Process (Sentiment)
    review_df = process_reviews(review_df)
    
    # 3. Feature Engineering (Merge reviews into business features)
    final_df = feature_engineering(business_df, review_df)
    print(f"Final feature set shape: {final_df.shape}")
    
    # 4. Model Training & Evaluation
    y_test, predictions, _ = train_and_evaluate(final_df)
    
    # 5. Visualization
    plot_results(y_test, predictions)
    
    print("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
