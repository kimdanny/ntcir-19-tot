import json
import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import random # Added for dummy data generation

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- Input Files ---
# Please replace the filenames in this list with your own JSONL files.
INPUT_FILES = [
    "YOUR_INPUT_FILE.jsonl"
]

# --- Output Files ---
# The names for the generated image and statistics files.
OUTPUT_IMAGE_FILE = "correlation_analysis.png"
OUTPUT_STATS_FILE = "correlation_stats.txt"

# --- New Change: Add a sampling fraction for visualization ---
# This will randomly sample a fraction of the data points for plotting to make the scatter plot clearer.
# 0.1 means 10% of the data will be plotted. Set to 1.0 to plot everything.
SAMPLE_FRACTION = 1.0

# ==============================================================================
# CORE FUNCTIONS
# ==============================================================================

def load_data_from_jsonl(filenames: list) -> pd.DataFrame:
    """Reads all specified JSONL files and extracts 'popularity' and 'doc_length'."""
    data_list = []
    
    for filename in filenames:
        if not os.path.exists(filename):
            print(f"Warning: Input file '{filename}' not found, skipping.")
            continue
        
        print(f"Reading data from '{filename}'...")
        with open(filename, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"Processing {os.path.basename(filename)}"):
                try:
                    data = json.loads(line)
                    # Only keep entries that have both key fields.
                    if 'popularity' in data and 'doc_length' in data and data['popularity'] > 0 and data['doc_length'] > 0:
                        data_list.append({
                            'popularity': data['popularity'],
                            'doc_length': data['doc_length']
                        })
                except json.JSONDecodeError:
                    print(f"Warning: Skipping a corrupted JSON line in '{filename}'.")
    
    if not data_list:
        return pd.DataFrame() # Return an empty DataFrame if no data was loaded.

    return pd.DataFrame(data_list)

def analyze_and_visualize(df: pd.DataFrame):
    """Calculates multiple correlation coefficients and generates a scatter plot."""
    if df.empty or len(df) < 2:
        print("Error: Not enough data to perform analysis.")
        return

    # --- 1. Quantitative Analysis: Calculated on the FULL dataset for accuracy ---
    
    # Pearson: Measures the linear relationship.
    pearson_corr = df['doc_length'].corr(df['popularity'], method='pearson')
    
    # Spearman: Measures the monotonic relationship (rank-based). Less sensitive to outliers.
    spearman_corr = df['doc_length'].corr(df['popularity'], method='spearman')
    
    # Kendall's Tau: Also measures the rank-based relationship, often used for smaller datasets.
    kendall_corr = df['doc_length'].corr(df['popularity'], method='kendall')
    
    print(f"\n--- Analysis Results (on full dataset) ---")
    print(f"Calculated Pearson Correlation: {pearson_corr:.4f}")
    print(f"Calculated Spearman Correlation: {spearman_corr:.4f}")
    print(f"Calculated Kendall's Tau: {kendall_corr:.4f}")
    
    # Write the results to a text file
    with open(OUTPUT_STATS_FILE, 'w', encoding='utf-8') as f:
        f.write("Analysis of the Relationship Between Document Length and Popularity\n")
        f.write("="*60 + "\n\n")
        f.write("General Interpretation of Correlation Coefficients:\n")
        f.write("  - Values are between -1 and +1.\n")
        f.write("  - Positive values (e.g., +0.3, +0.7) suggest a positive correlation (as one increases, the other tends to increase).\n")
        f.write("  - Negative values (e.g., -0.3, -0.7) suggest a negative correlation (as one increases, the other tends to decrease).\n")
        f.write("  - Values near 0 (e.g., -0.1 to +0.1) suggest a very weak or no correlation.\n")
        f.write("  - Strength: |0.1|-|0.3| is weak, |0.4|-|0.6| is moderate, |0.7|-|1.0| is strong.\n\n")
        
        f.write(f"Pearson Correlation Coefficient: {pearson_corr:.4f}\n")
        f.write("  - Specifically measures the LINEAR relationship.\n\n")
        
        f.write(f"Spearman's Rank Correlation: {spearman_corr:.4f}\n")
        f.write("  - Measures the MONOTONIC relationship (general positive/negative trends).\n\n")

        f.write(f"Kendall's Tau Correlation: {kendall_corr:.4f}\n")
        f.write("  - Also measures the strength of the monotonic relationship.\n")
    print(f"Quantitative analysis results saved to: '{OUTPUT_STATS_FILE}'")


    # --- 2. Visualization: Generated on a SAMPLE of the data for clarity ---
    print(f"\nGenerating visualization chart using a {SAMPLE_FRACTION*100:.0f}% sample of the data...")
    
    # Filter out zero values for log scale
    plot_df = df[(df['popularity'] > 0) & (df['doc_length'] > 0)].copy()

    # --- New Change: Sample the DataFrame before plotting ---
    if 0 < SAMPLE_FRACTION < 1.0:
        plot_df = plot_df.sample(frac=SAMPLE_FRACTION, random_state=42)
        print(f"Plotting {len(plot_df)} sampled data points.")

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 8))
    
    # Use regplot to create a scatter plot with a linear regression line
    ax = sns.regplot(
        x='doc_length',
        y='popularity',
        data=plot_df,
        scatter_kws={'alpha': 0.3, 's': 15}, # Slightly increased alpha and size for better visibility
        line_kws={'color': 'red', 'linewidth': 2} 
    )
    
    # Use a logarithmic scale to better visualize the distribution
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.set_title('Document Length vs. Popularity (Logarithmic Scale)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Document Length (Number of Characters) - Log Scale', fontsize=12)
    ax.set_ylabel('Popularity (Pageviews) - Log Scale', fontsize=12)
    
    # Annotate with the correlation calculated from the FULL dataset
    plt.text(0.05, 0.95, f'Pearson Correlation (Full Data): {pearson_corr:.4f}', 
             transform=ax.transAxes, fontsize=12, verticalalignment='top', 
             bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE_FILE, dpi=150)
    print(f"Visualization chart saved to: '{OUTPUT_IMAGE_FILE}'")
    plt.close()


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    if "dummy_enriched_data.jsonl" in INPUT_FILES and not os.path.exists("dummy_enriched_data.jsonl"):
        print("Creating dummy data file 'dummy_enriched_data.jsonl'...")
        with open("dummy_enriched_data.jsonl", 'w', encoding='utf-8') as f:
            for i in range(5000):
                entry = {
                    "id_ko": str(i), "title_ko": f"Sample Title {i}", "text_ko": "..." * i,
                    "popularity": int(max(1, 10000 / (i + 1) + random.randint(-200, 200))), 
                    "doc_length": 3 * i, "domains": ["example"]
                }
                f.write(json.dumps(entry) + '\n')
        print("Dummy data file created.")

    # 1. Load the data
    data_df = load_data_from_jsonl(INPUT_FILES)
    
    # 2. Analyze and visualize
    analyze_and_visualize(data_df)

