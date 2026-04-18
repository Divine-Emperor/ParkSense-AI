"""
ParkSense AI - Visualization Script
Generates comparison plots from the results.json file.
"""
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def main():
    try:
        with open("results.json", "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        print("No results.json found. Please run train_model.py first.")
        return

    sns.set_theme(style="whitegrid")
    
    # Extract data
    models = list(results.keys())
    accs = [results[m]["accuracy"] for m in models]
    mf1s = [results[m]["macro_f1"] for m in models]
    
    # Filter for models that have ECE/TCE
    prob_models = [m for m in models if results[m].get("ece") is not None]
    eces = [results[m]["ece"] for m in prob_models]
    tces = [results[m]["tce"] for m in prob_models]

    # Plot 1: Accuracy vs Macro-F1
    plt.figure(figsize=(10, 6))
    x = np.arange(len(models))
    width = 0.35
    
    plt.bar(x - width/2, accs, width, label='Accuracy', color='skyblue')
    plt.bar(x + width/2, mf1s, width, label='Macro-F1', color='salmon')
    
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, models, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300)
    print("Saved performance_comparison.png")

    # Plot 2: Calibration and Temporal Consistency
    if prob_models:
        plt.figure(figsize=(10, 6))
        x_prob = np.arange(len(prob_models))
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        color = 'tab:blue'
        ax1.set_xlabel('Models')
        ax1.set_ylabel('ECE (Lower is better)', color=color)
        ax1.bar(x_prob - width/2, eces, width, label='ECE', color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        plt.xticks(x_prob, prob_models, rotation=45)
        
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('TCE (Lower is better)', color=color)
        ax2.bar(x_prob + width/2, tces, width, label='TCE', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('Calibration (ECE) and Temporal Consistency (TCE)')
        fig.tight_layout()
        plt.savefig('calibration_tce.png', dpi=300)
        print("Saved calibration_tce.png")

if __name__ == "__main__":
    main()
