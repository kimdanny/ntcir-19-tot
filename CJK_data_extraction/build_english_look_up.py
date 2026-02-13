import json
import os
from tqdm import tqdm
from datasets import load_dataset

BASE_DIR = "YOUR_BASE_DIRECTORY_PATH"

INPUT_FILES = [
    "wiki_ko_only.jsonl",
    "wiki_zh_only.jsonl",
    "wiki_ja_only.jsonl",
    "wiki_zh_en_bilingual.jsonl",
    "wiki_ja_en_bilingual.jsonl",
    "wiki_ko_en_bilingual.jsonl",
]

OUTPUT_FILE = os.path.join(BASE_DIR, "en_lookup.jsonl")

DATASET_PATH = "YOUR_DATASET_PATH_HERE"
DATASET_CONFIG_EN = "20231101.en"


def create_english_lookup_table():
    """
    Scans CJK bilingual files to find all unique English article IDs,
    then extracts these full articles from the English Wikipedia dataset
    into a single, deduplicated JSONL file.
    """
    print("--- Step 1: Aggregating all unique English IDs from CJK files ---")
    required_english_ids = set()

    for filename in INPUT_FILES:
        input_path = os.path.join(BASE_DIR, filename)
        if not os.path.exists(input_path):
            print(f"Warning: Input file not found at '{input_path}'. Skipping.")
            continue

        print(f"Scanning '{filename}' for English IDs...")
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'id_en' in data and data['id_en']:
                        required_english_ids.add(data['id_en'])
                except json.JSONDecodeError:
                    pass

    if not required_english_ids:
        print("Error: No English IDs were found in the input files. Exiting.")
        return

    print(f"\nFound a total of {len(required_english_ids)} unique English articles to extract.")

    print("\n--- Step 2: Loading full English Wikipedia dataset and filtering... ---")
    print("Note: Loading the dataset may take several minutes and consume significant memory.")

    try:
        ds_en_full = load_dataset(DATASET_PATH, DATASET_CONFIG_EN, split='train')
    except Exception as e:
        print(f"Error loading English dataset: {e}")
        print("Please ensure the dataset path and configuration are correct.")
        return

    print("Filtering and writing English articles to the lookup table...")
    articles_written = 0
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for item_en in tqdm(ds_en_full, desc="Extracting English articles"):
            if item_en['id'] in required_english_ids:
                # Write the full article as a JSON line
                # Use the original field names from the HF dataset (id, url, title, text)
                f_out.write(json.dumps(item_en, ensure_ascii=False) + '\n')
                articles_written += 1
    
    print("\n--- Task Complete! ---")
    print(f"Successfully wrote {articles_written} articles to '{OUTPUT_FILE}'.")
    if articles_written < len(required_english_ids):
        print(f"Warning: {len(required_english_ids) - articles_written} IDs were not found in the English dataset.")


if __name__ == "__main__":
    create_english_lookup_table()