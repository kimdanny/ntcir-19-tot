# process_wiki_zh.py
import os
import json
import requests
import getpass
import urllib.parse
from typing import Optional, Set, List, Dict
from datasets import load_dataset
from tqdm import tqdm
import sqlite3 

# --- Configuration for Chinese ---
DATASET_CONFIG_ZH = "20231101.zh"
DATABASE_FILE = "/artifacts/english_wikipedia.db"
# --- Change 1: Add a constant for batch size ---
BATCH_SIZE = 50  # Number of titles to query in a single API call (max is 50)

# --- API Headers ---
HEADERS = {
    'User-Agent': 'Wikipedia processing script for academic research (contact: YOUR_EMAIL@EMAIL.COM)'
}

# --- Output Path Configuration ---
OUTPUT_DIR = "YOUR_OUTPUT_DIRECTORY_PATH"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_ZH_ONLY = os.path.join(OUTPUT_DIR, "wikipedia_zh_only.jsonl")
OUTPUT_BILINGUAL = os.path.join(OUTPUT_DIR, "wikipedia_bilingual_zh_en.jsonl")


# --- Testing Configuration ---
# Set to an integer (e.g., 100) for testing, or None for a full run.
NUM_SAMPLES_TO_TEST = None

# ==============================================================================
# Change 2: Replace the single-item function with a batch-processing function
# ==============================================================================
def get_english_links_batch(chinese_titles: List[str]) -> Dict[str, Optional[str]]:
    """
    Finds English titles for a given BATCH of Chinese titles in a single API call.
    Returns a dictionary mapping each Chinese title to its English counterpart.
    """
    titles_param = "|".join(chinese_titles)
    encoded_titles = urllib.parse.quote(titles_param)
    
    # Use the Chinese Wikipedia API endpoint with multiple titles
    url = f"https://zh.wikipedia.org/w/api.php?action=query&prop=langlinks&titles={encoded_titles}&lllang=en&format=json&redirects"

    # Initialize a result map with None for all requested titles
    results_map = {title: None for title in chinese_titles}

    try:
        response = requests.get(url, headers=HEADERS, timeout=30) # Increased timeout for larger requests
        response.raise_for_status()
        data = response.json()
        
        # The API can normalize or redirect titles, we need to map them back
        redirect_map = {item['from']: item['to'] for item in data.get("query", {}).get("redirects", [])}
        normalized_map = {item['from']: item['to'] for item in data.get("query", {}).get("normalized", [])}

        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            chinese_title_from_api = page.get('title')
            if not chinese_title_from_api:
                continue
            
            english_link = None
            if "langlinks" in page and page.get("langlinks"):
                english_link = page["langlinks"][0].get("*")

            # Store the result for the title returned by the API
            if chinese_title_from_api in results_map:
                results_map[chinese_title_from_api] = english_link

        # Map results from redirected/normalized titles back to the original titles
        for original_title in chinese_titles:
            final_title = normalized_map.get(original_title, original_title)
            final_title = redirect_map.get(final_title, final_title)
            
            if final_title != original_title and final_title in results_map:
                results_map[original_title] = results_map[final_title]

        return results_map

    except requests.exceptions.RequestException as e:
        print(f"    -> Batch API Error for titles starting with '{chinese_titles[0]}': {e}")
        return results_map # Return the map with Nones on error

# --- Load the processed titles ---
def load_processed_titles(zh_only_path: str, bilingual_path: str) -> Set[str]:
    """Reads existing output files and builds a set of all processed Chinese article titles."""
    processed_titles = set()
    for filepath in [zh_only_path, bilingual_path]:
        if not os.path.exists(filepath):
            continue
        print(f"Loading processed Chinese titles from {filepath}...")
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'title_zh' in data:
                        processed_titles.add(data['title_zh'])
                except json.JSONDecodeError:
                    pass  # Corrupted line, skip
    return processed_titles

def main():
    print(f"All output files will be saved to: {OUTPUT_DIR}")

    # Load already processed Chinese titles to avoid duplication
    processed_zh_titles = load_processed_titles(OUTPUT_ZH_ONLY, OUTPUT_BILINGUAL)
    if processed_zh_titles:
        print(f"Resuming job. Found {len(processed_zh_titles)} already processed Chinese titles. They will be skipped.")

    if not os.path.exists(DATABASE_FILE):
        print(f"Database file not found at '{DATABASE_FILE}'. Building it now...")
        print("This is a one-time setup and may take a significant amount of time.")
        
        
        DATASET_CONFIG_EN = "20231101.en"
        ds_en_full_build = load_dataset("/dataset/", DATASET_CONFIG_EN, split='train')
        
        
        con_build = sqlite3.connect(DATABASE_FILE)
        cur_build = con_build.cursor()
        cur_build.execute('''
            CREATE TABLE IF NOT EXISTS articles (
                title_lower TEXT PRIMARY KEY,
                article_data TEXT NOT NULL
            )
        ''')

        def item_generator():
            for item in ds_en_full_build:
                yield (item['title'].lower(), json.dumps(item))
        
        cur_build.executemany("INSERT OR IGNORE INTO articles (title_lower, article_data) VALUES (?, ?)", 
                        tqdm(item_generator(), total=len(ds_en_full_build), desc="Building DB"))

        con_build.commit()

        print("Creating index for fast lookups...")
        cur_build.execute("CREATE INDEX IF NOT EXISTS idx_title_lower ON articles (title_lower)")
        con_build.commit()
        con_build.close()
        print("Database build complete.")
    else:
        print(f"Found existing database at '{DATABASE_FILE}'. Skipping build step.")

    print(f"Loading Chinese Wikipedia dataset ({DATASET_CONFIG_ZH})...")
    ds_zh_full = load_dataset("/dataset/", DATASET_CONFIG_ZH, split='train')

    if NUM_SAMPLES_TO_TEST and NUM_SAMPLES_TO_TEST > 0:
        print(f"--- WARNING: Running in test mode, processing only the first {NUM_SAMPLES_TO_TEST} samples. ---")
        ds_zh = ds_zh_full.take(NUM_SAMPLES_TO_TEST)
    else:
        print("--- INFO: Running in full mode, processing all samples. ---")
        ds_zh = ds_zh_full


    con = sqlite3.connect(DATABASE_FILE)
    cur = con.cursor()

    # ==============================================================================
    # Change 3: Refactor the main loop to use batching
    # ==============================================================================
    batch = [] # A list to hold items for the next API call
    with open(OUTPUT_ZH_ONLY, "a", encoding="utf-8") as f_zh_only, \
         open(OUTPUT_BILINGUAL, "a", encoding="utf-8") as f_bilingual:
        print("Starting to process the Chinese dataset and classify articles...")
        
        # Manually manage tqdm for accurate progress tracking with batching
        progress_bar = tqdm(total=len(ds_zh_full), desc="Processing Chinese articles")
        progress_bar.update(len(processed_zh_titles)) # Set initial progress

        for item_zh in ds_zh_full:
            chinese_title = item_zh['title']
            if chinese_title in processed_zh_titles:     # Check if already processed
                continue
            
            # Add item to the current batch
            batch.append(item_zh)

            # When the batch is full, process it
            if len(batch) >= BATCH_SIZE:
                titles_to_query = [item['title'] for item in batch]
                # Make a single API call for the entire batch
                english_results = get_english_links_batch(titles_to_query)

                # Process each item in the batch with the results
                for item_in_batch in batch:
                    title_in_batch = item_in_batch['title']
                    english_title = english_results.get(title_in_batch)

                    if english_title:
                        cur.execute("SELECT article_data FROM articles WHERE title_lower = ?", (english_title.lower(),))
                        result = cur.fetchone()
                    else:
                        result = None

                    if result:
                        item_en = json.loads(result[0])
                        bilingual_record = {
                            "id_zh": item_in_batch['id'], "url_zh": item_in_batch['url'],
                            "title_zh": title_in_batch, "text_zh": item_in_batch['text'],
                            "id_en": item_en['id'],
                        }
                        f_bilingual.write(json.dumps(bilingual_record, ensure_ascii=False) + "\n")
                    else:
                        zh_only_record = {
                            "id_zh": item_in_batch['id'], "url_zh": item_in_batch['url'],
                            "title_zh": title_in_batch, "text_zh": item_in_batch['text'],
                        }
                        f_zh_only.write(json.dumps(zh_only_record, ensure_ascii=False) + "\n")
                
                progress_bar.update(len(batch)) # Update the progress bar
                batch = [] # Reset the batch

        # Process the final, potentially incomplete, batch
        if batch:
            titles_to_query = [item['title'] for item in batch]
            english_results = get_english_links_batch(titles_to_query)
            for item_in_batch in batch:
                title_in_batch = item_in_batch['title']
                english_title = english_results.get(title_in_batch)
                if english_title:
                    cur.execute("SELECT article_data FROM articles WHERE title_lower = ?", (english_title.lower(),))
                    result = cur.fetchone()
                else:
                    result = None
                if result:
                    item_en = json.loads(result[0])
                    bilingual_record = {"id_zh": item_in_batch['id'], "url_zh": item_in_batch['url'], "title_zh": title_in_batch, "text_zh": item_in_batch['text'], "id_en": item_en['id']}
                    f_bilingual.write(json.dumps(bilingual_record, ensure_ascii=False) + "\n")
                else:
                    zh_only_record = {"id_zh": item_in_batch['id'], "url_zh": item_in_batch['url'], "title_zh": title_in_batch, "text_zh": item_in_batch['text']}
                    f_zh_only.write(json.dumps(zh_only_record, ensure_ascii=False) + "\n")
            progress_bar.update(len(batch))

        progress_bar.close()

    con.close()
    print("\nProcessing complete!")
    print(f"Output files have been saved to the directory: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()

