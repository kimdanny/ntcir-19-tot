import json
import os
import random
from tqdm import tqdm
# --- New Dependencies: Install matplotlib and seaborn for plotting ---
# You can install them by running: pip install matplotlib seaborn
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- Input files ---
# Please replace the filenames in this list with your own JSONL filenames.
INPUT_FILES = [
    "YOUR_INPUT_FILE_1.jsonl",
    "YOUR_INPUT_FILE_2.jsonl",
]

# --- Output file ---
# The script will now generate a PNG image file.
OUTPUT_IMAGE_FILE = "distribution_plots.png"

# --- BINNING RULES ---
# Define the bin edges for 'popularity'.
# The first bin will be 100-499, then 500-999, etc.
POPULARITY_BINS = [100] + list(range(500, 5501, 500)) + list(range(6000, 10001, 1000)) + [12000, 15000, 20000, 30000, 50000, 100000, float('inf')]

# Define the bin edges for 'doc_length'.
# 0-5k every 500, 5k-10k every 1000, then larger bins.
DOC_LENGTH_BINS = [200] + list(range(500, 5501, 500)) + list(range(6000, 10001, 1000)) + [12000, 15000, 20000, 30000, 50000, 100000, float('inf')]


# ==============================================================================
# CORE FUNCTIONS
# ==============================================================================

def read_and_extract_data(filenames):
    """Reads all specified JSONL files and extracts 'popularity' and 'doc_length'."""
    all_popularities = []
    all_doc_lengths = []
    
    for filename in filenames:
        if not os.path.exists(filename):
            print(f"Warning: Input file '{filename}' not found, skipping.")
            continue
        
        print(f"Reading data from '{filename}'...")
        with open(filename, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"Processing {os.path.basename(filename)}"):
                try:
                    data = json.loads(line)
                    if 'popularity' in data:
                        all_popularities.append(data['popularity'])
                    if 'doc_length' in data:
                        all_doc_lengths.append(data['doc_length'])
                except json.JSONDecodeError:
                    print(f"Warning: Skipping a corrupted JSON line in '{filename}'.")
                    
    return all_popularities, all_doc_lengths

def bin_data(data, bin_edges):
    """Bins and counts data according to the given bin edges."""
    counts = [0] * (len(bin_edges) - 1)
    
    for value in data:
        for i in range(len(bin_edges) - 1):
            if bin_edges[i] <= value < bin_edges[i+1]:
                counts[i] += 1
                break
                
    labels = []
    for i in range(len(bin_edges) - 1):
        start = bin_edges[i]
        end = bin_edges[i+1]
        if end == float('inf'):
            labels.append(f"{start:,}+")
        else:
            # -1 to make the range inclusive, e.g., 0-499
            labels.append(f"{start:,} - {end-1:,}")
            
    return {'labels': labels, 'counts': counts}

# --- New Change: Replaced the HTML generation with a simpler image plotting function ---
def create_plot_image(pop_data, len_data, output_filename):
    """Generates a single PNG image file containing two histograms."""
    
    print(f"\nGenerating plot image: '{output_filename}'...")
    
    # Set the aesthetic style of the plots
    sns.set_theme(style="whitegrid")
    
    # Create a figure and a set of subplots (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('Korean Wikipedia Data Distribution', fontsize=24, fontweight='bold')

    # --- Plot 1: Popularity Distribution ---
    pop_palette = sns.color_palette("coolwarm", len(pop_data['labels']))
    sns.barplot(ax=axes[0], x=pop_data['labels'], y=pop_data['counts'], palette=pop_palette, hue=pop_data['labels'], dodge=False, legend=False)
    axes[0].set_title('Wiki Entities Popularity Distribution', fontsize=16)
    axes[0].set_xlabel('Popularity (Pageviews)', fontsize=12)
    axes[0].set_ylabel('Number of Entities (Count)', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45, labelsize=10)
    # Add text labels on top of each bar
    for i, count in enumerate(pop_data['counts']):
        if count > 0:
            axes[0].text(i, count, f'{count:,}', ha='center', va='bottom', fontsize=9)


    # --- Plot 2: Document Length Distribution ---
    len_palette = sns.color_palette("viridis", len(len_data['labels']))
    sns.barplot(ax=axes[1], x=len_data['labels'], y=len_data['counts'], palette=len_palette, hue=len_data['labels'], dodge=False, legend=False)
    axes[1].set_title('Document Length Distribution', fontsize=16)
    axes[1].set_xlabel('Document Length (Characters)', fontsize=12)
    axes[1].set_ylabel('') # No need for a second y-axis label
    axes[1].tick_params(axis='x', rotation=45, labelsize=10)
    # Add text labels on top of each bar
    for i, count in enumerate(len_data['counts']):
        if count > 0:
            axes[1].text(i, count, f'{count:,}', ha='center', va='bottom', fontsize=9)

    # Adjust layout to prevent labels from overlapping
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the figure to a file
    plt.savefig(output_filename, dpi=150)
    print(f"Plot saved successfully to '{output_filename}'")
    plt.close()


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    # Check if dummy files exist, create them if not.
    # if not all(os.path.exists(f) for f in INPUT_FILES if "dummy" in f):
    #     print("Creating dummy data files...")
    #     # Create dummy data if not present
    #     for i in range(1, 3):
    #         filename = f"dummy_data_{i}.jsonl"
    #         with open(filename, 'w', encoding='utf-8') as f:
    #             for j in range(10000): # Create 10,000 sample entries
    #                 entry = {
    #                     f"id_ko": str(j),
    #                     f"title_ko": f"Sample Title {j}",
    #                     "popularity": int(random.lognormvariate(8, 2)), # Log-normal distribution
    #                     "doc_length": int(random.lognormvariate(7, 1.5))
    #                 }
    #                 if i == 2: # Add id_en to the second file
    #                     entry["id_en"] = str(j + 10000)
    #                 f.write(json.dumps(entry) + '\n')
    #     print("Dummy data files created.")


    # 1. Read and extract data
    popularities, doc_lengths = read_and_extract_data(INPUT_FILES)
    
    if not popularities and not doc_lengths:
        print("Error: Could not read any data from the input files. Exiting.")
    else:
        # 2. Bin the data
        print("\nBinning the data...")
        binned_popularity = bin_data(popularities, POPULARITY_BINS)
        binned_length = bin_data(doc_lengths, DOC_LENGTH_BINS)
        
        # 3. Create the plot image report
        create_plot_image(binned_popularity, binned_length, OUTPUT_IMAGE_FILE)

