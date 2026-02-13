import json
import os
from datasets import load_dataset
from tqdm import tqdm

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Path to the locally cached/mounted Hugging Face datasets
# This should match the path used in your VESSL YAML configuration.
DATASET_PATH = "YOUR_DATASET_PATH_HERE"

# The specific version of the English Wikipedia dataset to load.
DATASET_CONFIG_EN = "20231101.en"

# Directory where the output file will be saved.
OUTPUT_DIR = "YOUR_OUTPUT_DIRECTORY_PATH"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# The name of the final output file.
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "wikipedia_en_metadata_only.jsonl")

# ==============================================================================
# MAIN SCRIPT
# ==============================================================================

def create_metadata_file():
    """
    Loads the full English Wikipedia dataset and writes a new JSONL file
    containing only the metadata (id, url, title) for each article,
    excluding the 'text' field.
    
    Note: This script is memory-intensive as it loads the entire dataset.
    It should be run on an instance with sufficient RAM (e.g., 32GB+).
    """
    
    print("--- Starting Wikipedia Metadata Extraction ---")
    
    # --- Step 1: Load the full English Wikipedia dataset ---
    print(f"Loading dataset 'wikimedia/wikipedia' with config '{DATASET_CONFIG_EN}'...")
    try:
        ds_en_full = load_dataset(DATASET_PATH, DATASET_CONFIG_EN, split='train')
        print(f"Successfully loaded {len(ds_en_full)} articles.")
    except Exception as e:
        print(f"FATAL: Failed to load the dataset. Error: {e}")
        print("Please ensure the DATASET_PATH and DATASET_CONFIG_EN are correct.")
        return

    # --- Step 2: Iterate, process, and write to the output file ---
    print(f"Writing metadata to: {OUTPUT_FILE}")
    
    # Open the output file in write mode
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        # Loop through each article in the dataset with a progress bar
        for item in tqdm(ds_en_full, desc="Processing articles"):
            # Create a new dictionary containing only the desired metadata fields.
            # This explicitly excludes the 'text' field.
            metadata_item = {
                'id': item['id'],
                'url': item['url'],
                'title': item['title']
            }
            
            # Convert the dictionary to a JSON string and write it as a new line.
            f_out.write(json.dumps(metadata_item, ensure_ascii=False) + '\n')
            
    print("\n--- Process complete! ---")
    print(f"Successfully created metadata file at: {OUTPUT_FILE}")


if __name__ == "__main__":
    create_metadata_file()
