import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_results(y_test, predictions, plots_dir="plots"):
    print("Generating plots...")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        
    # 1. Actual vs Predicted (Random Forest)
    y_pred = predictions['Random Forest']
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([1, 5], [1, 5], 'r--')
    plt.xlabel("Actual Ratings")
    plt.ylabel("Predicted Ratings")
    plt.title("Actual vs Predicted Ratings (Random Forest)")
    plt.savefig(f"{plots_dir}/actual_vs_predicted.png")
    plt.close()
    
    # 2. Residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True)
    plt.title("Residual Distribution")
    plt.xlabel("Error (Actual - Predicted)")
    plt.savefig(f"{plots_dir}/residuals.png")
    plt.close()
