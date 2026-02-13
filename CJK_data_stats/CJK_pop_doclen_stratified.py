import json
import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import random

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- Input Files ---
INPUT_FILES = [
    "YOUR_INPUT_FILE_1.jsonl"
]

# --- Output Files ---
OUTPUT_STATS_FILE = "correlation_stats.txt"
OUTPUT_IMAGE_STRATIFIED_FILE = "stratified_correlation.png" # For stratified visualization
OUTPUT_IMAGE_OVERALL_FILE = "overall_correlation.png" # For overall visualization

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
                    if 'popularity' in data and 'doc_length' in data and data['popularity'] > 0 and data['doc_length'] > 0:
                        data_list.append({
                            'popularity': data['popularity'],
                            'doc_length': data['doc_length']
                        })
                except json.JSONDecodeError:
                    pass # Skip corrupted lines
    
    if not data_list:
        return pd.DataFrame()
    return pd.DataFrame(data_list)

def analyze_and_visualize(df: pd.DataFrame):
    """Performs overall and stratified correlation analysis and generates visualizations."""
    if df.empty or len(df) < 2:
        print("Error: Not enough data for analysis.")
        return

    # --- 1. Macro Overall Analysis ---
    pearson_corr = df['doc_length'].corr(df['popularity'], method='pearson')
    spearman_corr = df['doc_length'].corr(df['popularity'], method='spearman')
    kendall_corr = df['doc_length'].corr(df['popularity'], method='kendall')

    with open(OUTPUT_STATS_FILE, 'w', encoding='utf-8') as f:
        f.write("Analysis of the Relationship Between Document Length and Popularity\n")
        f.write("="*60 + "\n\n")
        f.write("### Macro Overall Analysis ###\n")
        f.write("This is the single correlation calculated on all data points together, reflecting the overall, cross-group trend.\n")
        f.write(f"Pearson Correlation: {pearson_corr:.4f}\n")
        f.write(f"Spearman Correlation: {spearman_corr:.4f}\n")
        f.write(f"Kendall Correlation: {kendall_corr:.4f}\n")

    print(f"Macro Overall analysis results have been written to: '{OUTPUT_STATS_FILE}'")

    # --- 2. Stratified Quantitative Analysis ---
    print("\n--- Performing stratified analysis... ---")
    
    try:
        df['length_stratum'] = pd.qcut(
            df['doc_length'], 
            q=5, 
            labels=["1st 20%", "2nd 20%", "3rd 20%", "4th 20%", "5th 20%"],
            duplicates='drop'
        )
    except ValueError as e:
        print(f"Could not create 10 unique strata, data distribution may be too concentrated: {e}")
        df['length_stratum'] = "All Data"

    # Prepare lists to store stratum-level results
    stratum_pearsons = []
    stratum_spearmans = []
    stratum_kendalls = []

    # Calculate correlations for each stratum
    with open(OUTPUT_STATS_FILE, 'a', encoding='utf-8') as f:
        f.write("\n\n### Stratified Analysis by Document Length ###\n")
        f.write("-" * 60 + "\n")
        
        for stratum_name, stratum_df in df.groupby('length_stratum', observed=True):
            print(f"Analyzing stratum: {stratum_name} (contains {len(stratum_df)} samples)")
            if len(stratum_df) < 2:
                print("  -> Too few samples, skipping analysis.")
                continue
                
            s_pearson = stratum_df['doc_length'].corr(stratum_df['popularity'], method='pearson')
            s_spearman = stratum_df['doc_length'].corr(stratum_df['popularity'], method='spearman')
            s_kendall = stratum_df['doc_length'].corr(stratum_df['popularity'], method='kendall')

            # Append the results from this stratum to our lists
            stratum_pearsons.append(s_pearson)
            stratum_spearmans.append(s_spearman)
            stratum_kendalls.append(s_kendall)

            f.write(f"Stratum: {stratum_name} (n={len(stratum_df)})\n")
            f.write(f"  - Range: {stratum_df['doc_length'].min():.0f} to {stratum_df['doc_length'].max():.0f} characters\n")
            f.write(f"  - Pearson:  {s_pearson:.4f}\n")
            f.write(f"  - Spearman: {s_spearman:.4f}\n")
            f.write(f"  - Kendall:  {s_kendall:.4f}\n\n")

        # Calculate and write the Micro Overall Analysis
        if stratum_pearsons: # Check if we have any results to average
            micro_pearson = sum(stratum_pearsons) / len(stratum_pearsons)
            micro_spearman = sum(stratum_spearmans) / len(stratum_spearmans)
            micro_kendall = sum(stratum_kendalls) / len(stratum_kendalls)

            f.write("\n\n### Micro Overall Analysis (Average of Strata) ###\n")
            f.write("This is the arithmetic average of the correlation coefficients from each stratum.\n")
            f.write("It reflects the 'typical' local relationship strength after controlling for document length.\n")
            f.write("-" * 60 + "\n")
            f.write(f"Average Pearson Correlation: {micro_pearson:.4f}\n")
            f.write(f"Average Spearman Correlation: {micro_spearman:.4f}\n")
            f.write(f"Average Kendall Correlation: {micro_kendall:.4f}\n")
            
    print(f"Stratified and Micro analysis results have been appended to: '{OUTPUT_STATS_FILE}'")

    # --- 3. Visualization ---
    print("\n--- Generating visualizations... ---")
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(12, 8))
    ax_log = sns.regplot(
        x='doc_length', y='popularity', data=df.sample(frac=0.1, random_state=42),
        scatter_kws={'alpha': 0.2, 's': 10}, line_kws={'color': 'red'}
    )
    ax_log.set_xscale('log')
    ax_log.set_yscale('log')
    ax_log.set_title('Overall Relationship: Document Length vs. Popularity (Log Scale)', fontsize=16)
    ax_log.set_xlabel('Document Length (Characters) - Log Scale', fontsize=12)
    ax_log.set_ylabel('Popularity (Pageviews) - Log Scale', fontsize=12)
    plt.text(0.05, 0.95, f'Pearson (Full Data): {pearson_corr:.4f}', transform=ax_log.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
    plt.savefig(OUTPUT_IMAGE_OVERALL_FILE, dpi=150)
    plt.close()
    print(f"Overall relationship plot saved to: '{OUTPUT_IMAGE_OVERALL_FILE}'")

    g = sns.lmplot(
        data=df, 
        x='doc_length', 
        y='popularity',
        col='length_stratum',
        col_wrap=3,          
        height=5,            
        sharex=False,        
        sharey=False,        
        scatter_kws={'alpha': 0.2, 's': 10},
        line_kws={'color': 'red'}
    )
    
    g.set(xscale="log", yscale="log")
    g.fig.suptitle('Stratified Relationship: Doc Length vs. Popularity (by Length Strata)', fontsize=20, y=1.03)
    g.set_axis_labels("Document Length (Log Scale)", "Popularity (Log Scale)")

    plt.savefig(OUTPUT_IMAGE_STRATIFIED_FILE, dpi=150)
    plt.close()
    print(f"Stratified relationship plot saved to: '{OUTPUT_IMAGE_STRATIFIED_FILE}'")

if __name__ == "__main__":
    if "dummy_enriched_data.jsonl" in INPUT_FILES and not os.path.exists("dummy_enriched_data.jsonl"):
        print("Creating dummy data file...")
        with open("dummy_enriched_data.jsonl", 'w', encoding='utf-8') as f:
            for i in range(5000):
                entry = {"popularity": int(max(10, random.lognormvariate(8, 2))), "doc_length": int(max(10, random.lognormvariate(7, 1.5)))}
                f.write(json.dumps(entry) + '\n')
        print("Dummy data file created.")

    data_df = load_data_from_jsonl(INPUT_FILES)
    analyze_and_visualize(data_df)

