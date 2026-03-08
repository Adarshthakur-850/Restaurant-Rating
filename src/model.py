from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import numpy as np

def train_and_evaluate(df):
    print("Training models...")
    
    # Features
    features = ['price_range', 'feature_count', 'latitude', 'longitude', 
                'category_encoded', 'city_encoded', 
                'sent_mean', 'sent_min', 'sent_max', 'sent_std']
    
    # We might use 'review_count' or 'review_count_real'. Let's use real count.
    features = ['price_range', 'review_count_real', 'category_encoded', 'city_encoded', 'sent_mean']
    
    X = df[features]
    y = df['average_stars'] # Target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    
    best_model = None
    best_rmse = float('inf')
    predictions = {}
    
    print(f"{'Model':<20} {'MAE':<10} {'RMSE':<10} {'R2':<10}")
    print("-" * 50)
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"{name:<20} {mae:.4f}     {rmse:.4f}     {r2:.4f}")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            
    # Save best model
    if not os.path.exists("models"):
        os.makedirs("models")
    joblib.dump(best_model, "models/rating_model.pkl")
    print(f"\nBest model saved to models/rating_model.pkl")
    
    return y_test, predictions, X_test
