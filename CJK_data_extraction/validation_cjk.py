import csv
import os
import requests
import urllib.parse
from typing import Optional, Dict, List, Set
from tqdm import tqdm
from datasets import load_dataset

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Input TSV file containing the queries
INPUT_FILE = "YOUR_INPUT_FILE_PATH.tsv"

# Directory where the output TSV files will be saved
OUTPUT_DIR = "YOUR_OUTPUT_DIRECTORY_PATH"

# Path to the locally cached/mounted Hugging Face datasets
DATASET_PATH = "YOUR_DATASET_PATH_HERE"

# API headers for Wikipedia requests
WIKI_HEADERS = {
    'User-Agent': 'CJK Query Enrichment Script (contact: YOUR_EMAIL@EMAIL.COM)'
}

# Target language configuration
TARGET_LANGUAGES = {
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean"
}

# Batch size for Wikipedia API calls
BATCH_SIZE = 50 

# ==============================================================================
# CORE FUNCTIONS
# ==============================================================================

def build_hf_lookup_maps() -> Dict[str, Dict[str, Dict]]:
    """
    Loads all CJK Hugging Face datasets into memory to create fast lookup maps.
    This is the most memory-intensive part of the script.
    Maps: {lang_code: {title: {'id': hf_id, 'title': hf_title}}}
    """
    print("--- Building Hugging Face dataset lookup maps (this requires a lot of memory)... ---")
    hf_lookup_maps = {}
    for lang_code in TARGET_LANGUAGES.keys():
        config_name = f"20231101.{lang_code}"
        print(f"Loading dataset for '{lang_code}' ({config_name})...")
        try:
            ds_full = load_dataset(DATASET_PATH, config_name, split='train')
            lang_map = {item['title']: {'id': item['id'], 'title': item['title']} for item in ds_full}
            hf_lookup_maps[lang_code] = lang_map
            print(f"Finished building map for '{lang_code}' with {len(lang_map)} entries.")
        except Exception as e:
            print(f"FATAL: Failed to load dataset for '{lang_code}'. Error: {e}")
            # Exit if any dataset fails to load, as it's critical
            exit(1)
    return hf_lookup_maps

def get_cjk_links_batch(english_titles: List[str]) -> Dict[str, Dict[str, Optional[str]]]:
    """
    For a batch of English titles, finds all CJK translations by making a separate,
    efficient batch API call for each target language.
    This version incorporates the user's proven logic and corrects the URL encoding bug.
    Returns a map: {english_title: {zh: zh_title, ja: ja_title, ko: ko_title}}
    """
    print(f"Fetching CJK translations for {len(english_titles)} unique English titles via API...")
    
    # Initialize the final results map
    results_map = {title: {lang: None for lang in TARGET_LANGUAGES} for title in english_titles}
    
    # --- The Corrected Logic: One batch API call PER language ---
    for lang_code in TARGET_LANGUAGES.keys():
        print(f"  Querying for language: {lang_code}...")
        
        # Process all english_titles in chunks of BATCH_SIZE
        for i in tqdm(range(0, len(english_titles), BATCH_SIZE), desc=f"  API batches for {lang_code}"):
            batch_of_original_titles = english_titles[i:i + BATCH_SIZE]
            
            # --- THE CRITICAL FIX IS HERE ---
            # 1. Individually encode each title.
            # 2. Join them with the literal '|' character, which is NOT encoded.
            encoded_titles = "|".join([urllib.parse.quote(title) for title in batch_of_original_titles])
            
            url = (
                f"https://en.wikipedia.org/w/api.php?action=query&prop=langlinks"
                f"&titles={encoded_titles}&lllang={lang_code}"
                f"&format=json&redirects"
            )

            try:
                response = requests.get(url, headers=WIKI_HEADERS, timeout=30)
                response.raise_for_status()
                data = response.json()

                # --- Using your robust logic for handling redirects and normalization ---
                redirect_map = {item['from']: item['to'] for item in data.get("query", {}).get("redirects", [])}
                normalized_map = {item['from']: item['to'] for item in data.get("query", {}).get("normalized", [])}

                temp_results_for_api_titles = {}
                pages = data.get("query", {}).get("pages", {})
                for page in pages.values():
                    title_from_api = page.get('title')
                    if not title_from_api:
                        continue
                    
                    if "langlinks" in page and page.get("langlinks"):
                        temp_results_for_api_titles[title_from_api] = page["langlinks"][0].get('*')

                # --- Map results back to the original titles we requested ---
                for original_title in batch_of_original_titles:
                    final_title = normalized_map.get(original_title, original_title)
                    final_title = redirect_map.get(final_title, final_title)
                    
                    if final_title in temp_results_for_api_titles:
                        results_map[original_title][lang_code] = temp_results_for_api_titles[final_title]

            except requests.exceptions.RequestException as e:
                print(f"    -> Batch API Error for a batch starting with '{batch_of_original_titles[0]}': {e}")
                
    return results_map

def main():
    """Main processing loop."""
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file '{INPUT_FILE}' not found. Please check the path.")
        return

    # 1. Memory-intensive setup: Load all CJK datasets into lookup maps
    hf_lookup_maps = build_hf_lookup_maps()

    # 2. Collect unique English titles from the input TSV
    print("\n--- Reading input file to collect unique English titles... ---")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in, delimiter='\t')
        original_rows = list(reader)
        unique_english_titles = sorted(list({row['rel_doc_title'] for row in original_rows}))
        # print("[DEBUG] Sample unique English titles:", unique_english_titles[:10])
    
    # 3. Get all CJK translations for these titles in efficient batches
    cjk_translations = get_cjk_links_batch(unique_english_titles)
    # print("[DEBUG] Sample CJK translations:", {k: cjk_translations[k] for k in unique_english_titles[:10]})

    # 4. Prepare output files and write headers
    output_writers = {}
    for lang_code in TARGET_LANGUAGES.keys():
        filename = os.path.join(OUTPUT_DIR, f"{lang_code}_output.tsv")
        f = open(filename, 'w', encoding='utf-8', newline='')
        original_headers = original_rows[0].keys()
        new_headers = list(original_headers) + [f"{lang_code}_rel_doc_id", f"{lang_code}_rel_doc_title"]
        writer = csv.DictWriter(f, fieldnames=new_headers, delimiter='\t')
        writer.writeheader()
        output_writers[lang_code] = writer
    
    # 5. Enrich and write the data
    print("\n--- Enriching data and writing to output files... ---")
    for row in tqdm(original_rows, desc="Processing rows"):
        english_title = row['rel_doc_title']
        
        # Get the pre-fetched translations for this English title
        translations = cjk_translations.get(english_title, {})

        # Process for each target language
        for lang_code in TARGET_LANGUAGES.keys():
            translated_title = translations.get(lang_code)
            
            # print(f"[DEBUG] Processing '{english_title}' for language '{lang_code}': Translated title: '{translated_title}'")
            # print(f"[DEBUG] HF lookup map size for '{lang_code}': {len(hf_lookup_maps[lang_code])}")
            # Check if a translation was found by the API AND if that title exists in our HF dataset map.
            if translated_title and translated_title in hf_lookup_maps[lang_code]:
                
                # If the condition is met, get the HF data.
                hf_entry = hf_lookup_maps[lang_code][translated_title]
                hf_id = hf_entry['id']
                hf_title = hf_entry['title']

                # Create a copy of the original row.
                new_row = row.copy()
                
                # Add the new, successfully found fields.
                new_row[f"{lang_code}_rel_doc_id"] = hf_id
                new_row[f"{lang_code}_rel_doc_title"] = hf_title
                
                # Write the enriched row to the correct language file ONLY if the match was successful.
                output_writers[lang_code].writerow(new_row)

    print("\n--- All tasks complete! ---")
    for lang_code in TARGET_LANGUAGES.keys():
        print(f"Output saved to: {os.path.join(OUTPUT_DIR, f'{lang_code}_output.tsv')}")

if __name__ == "__main__":
    main()

