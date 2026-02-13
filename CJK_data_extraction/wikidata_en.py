import json
import os
from datasets import load_dataset
from tqdm import tqdm

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- Hugging Face Dataset Details ---
DATASET_CONFIG = "20231101.en"
DATASET_PATH = "YOUR_DATASET_PATH_HERE"

# --- Output Configuration ---
OUTPUT_DIR = "YOUR_OUTPUT_DIRECTORY_PATH"
OUTPUT_FILENAME = "wiki_en.jsonl"

# --- Scalability Setting ---
# Set to an integer (e.g., 1000) to download only the first N samples for testing.
# Set to None to download and process the entire dataset.
NUM_SAMPLES_TO_DOWNLOAD = None

# ==============================================================================
# CORE FUNCTIONS 
# ==============================================================================

def download_and_save_wikipedia():
    """
    Downloads the English Wikipedia dataset from Hugging Face,
    selects and renames specific columns, and saves it as a JSONL file.
    Supports downloading a subset or the full dataset based on NUM_SAMPLES_TO_DOWNLOAD.
    """
    
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    os.makedirs(OUTPUT_DIR, exist_ok=True) # Ensure output directory exists

    print("--- Starting English Wikipedia Dataset Download and Formatting ---")
    
    # --- Step 1: Load the dataset ---
    split_config = 'train'
    if NUM_SAMPLES_TO_DOWNLOAD is not None and NUM_SAMPLES_TO_DOWNLOAD > 0:
        print(f"Loading the first {NUM_SAMPLES_TO_DOWNLOAD} samples for testing...")
        # Efficiently load only the first N samples using split slicing
        split_config = f'train[:{NUM_SAMPLES_TO_DOWNLOAD}]'
    else:
        print(f"Loading the full dataset (this might take a while and require significant disk space)...")
        
    try:
        # Load the dataset (either subset or full)
        ds_en = load_dataset(DATASET_PATH, DATASET_CONFIG, split=split_config)
        
        # Get the total number of items to process for tqdm
        num_items = len(ds_en) if NUM_SAMPLES_TO_DOWNLOAD is None else NUM_SAMPLES_TO_DOWNLOAD

    except Exception as e:
        print(f"FATAL: Failed to load the dataset. Error: {e}")
        print("Please ensure the DATASET_PATH and DATASET_CONFIG are correct, and you have enough disk space.")
        return

    print(f"Successfully loaded {num_items} articles.")

    # --- Step 2: Process and write to JSONL file ---
    print(f"Processing articles and writing to: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f_out:
        # Iterate through the dataset with a progress bar
        for item in tqdm(ds_en, total=num_items, desc="Writing JSONL"):
            # Create a new dictionary with the desired fields and renamed keys
            formatted_item = {
                'id_en': item['id'],
                'url_en': item['url'],
                'title_en': item['title'],
                'text_en': item['text']
            }
            
            # Convert to JSON string and write to file
            f_out.write(json.dumps(formatted_item, ensure_ascii=False) + '\n')

    print("\n--- Process Complete! ---")
    print(f"Successfully saved formatted data to: {output_path}")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    download_and_save_wikipedia()
