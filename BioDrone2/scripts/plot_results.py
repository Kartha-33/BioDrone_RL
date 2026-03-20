import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set style for scientific publication
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("tab10")

def plot_benchmark():
    if not os.path.exists('results/benchmark_phase_E.csv'):
        print("Error: No results found at results/benchmark_phase_E.csv")
        return

    # Load Data
    df = pd.read_csv('results/benchmark_phase_E.csv')
    
    # Create Plot
    plt.figure(figsize=(10, 6))
    
    # Plot line with confidence interval (aggregated over seeds)
    sns.lineplot(
        data=df, 
        x='episode', 
        y='reward', 
        hue='model', 
        style='model',
        linewidth=2.5,
        alpha=0.8
    )
    
    plt.title('Performance: Dense vs. Bio-Constrained Architecture', fontsize=14, fontweight='bold')
    plt.xlabel('Training Episodes', fontsize=12)
    plt.ylabel('Total Reward', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title='Architecture', loc='upper left')
    
    # Save
    os.makedirs('plots', exist_ok=True)
    out_path = 'plots/learning_curve.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"✅ Success! Plot saved to {out_path}")

    # Print Summary Statistics
    print("\n--- Final Performance (Last 50 Episodes) ---")
    last_50 = df[df['episode'] > 450]
    summary = last_50.groupby('model')['reward'].agg(['mean', 'std']).reset_index()
    print(summary)

if __name__ == "__main__":
    plot_benchmark()