import json
import os
import requests
import urllib.parse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Base directory for input and output files
BASE_INPUT_DIR = "/en_datasets/"
BASE_OUTPUT_DIR = "/artifacts/"

# API Headers (please be sure to fill this in)
HEADERS = {
    'User-Agent': 'Data Enrichment Script (contact: YOUR_EMAIL@EMAIL.COM)'
}

# Date range for popularity data (format: YYYYMMDDHH)
START_DATE = "2023010100"
END_DATE = "2023110123"

# Number of parallel API request threads (can be adjusted based on your network environment)
MAX_WORKERS = 50

# ==============================================================================
# CORE FUNCTIONS
# ==============================================================================

def fetch_popularity(session, title, lang_code):
    """
    Fetches the total pageviews for a single article over a defined period.
    Returns the total view count, or 0 if not found or an error occurs.
    """
    if not title:
        return title, 0
    
    # URL-encode the article title to handle special characters
    encoded_title = urllib.parse.quote(title, safe='')
    
    # Build the Pageviews API URL
    url = (
        f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
        f"{lang_code}.wikipedia/all-access/user/{encoded_title}/monthly/{START_DATE}/{END_DATE}"
    )
    
    try:
        response = session.get(url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            data = response.json()
            # Sum the pageviews from all time points
            total_views = sum(item.get('views', 0) for item in data.get('items', []))
            return title, total_views
        else:
            return title, 0 # API returned an error (e.g., 404 Not Found)
    except requests.exceptions.RequestException:
        return title, 0 # Network request failed

def process_file(input_filename, output_filename, lang_code, title_key, text_key):
    """
    Main function to process a single JSONL file using a two-pass streaming method
    to avoid high memory usage.
    """
    input_path = os.path.join(BASE_INPUT_DIR, input_filename)
    output_path = os.path.join(BASE_OUTPUT_DIR, output_filename)

    if not os.path.exists(input_path):
        print(f"SKIPPING: Input file not found: {input_path}")
        return

    print(f"\n--- Starting to process file: {input_filename} ---")

    # --- Step 1: Read all titles (Streaming Pass 1) ---
    # We only store titles in memory, not the entire article content.
    print("--- Step 1: Reading all titles (Streaming Pass 1)... ---")
    titles = []
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Pass 1: Reading titles"):
                try:
                    article = json.loads(line)
                    titles.append(article.get(title_key))
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed JSON line in Pass 1")
    except FileNotFoundError:
        print(f"ERROR: Input file not found at {input_path}")
        return
        
    print(f"Found {len(titles)} titles.")

    # --- Step 2: Use multi-threading to fetch popularity for all articles in parallel ---
    # This part is unchanged and operates on the 'titles' list.
    print(f"--- Step 2: Fetching popularity data for {len(titles)} articles... ---")
    popularity_map = {}
    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_title = {executor.submit(fetch_popularity, session, title, lang_code): title for title in titles}
            
            for future in tqdm(as_completed(future_to_title), total=len(titles), desc=f"Fetching popularity for {lang_code}"):
                original_title, views = future.result()
                if original_title is not None:
                    popularity_map[original_title] = views

    # --- Step 3: Enrich data and write to a new file (Streaming Pass 2) ---
    # We re-read the input file line-by-line, enrich, and write immediately.
    print("--- Step 3: Enriching and writing new file (Streaming Pass 2)... ---")
    try:
        with open(output_path, 'w', encoding='utf-8') as f_out:
            with open(input_path, 'r', encoding='utf-8') as f_in:
                for line in tqdm(f_in, desc="Pass 2: Writing enriched data"):
                    try:
                        article = json.loads(line)
                        
                        # Get the popularity from the map
                        title = article.get(title_key)
                        article['popularity'] = popularity_map.get(title, 0)
                        
                        # Calculate and append the document length
                        article['doc_length'] = len(article.get(text_key, ""))
                        
                        # Write the new, enriched line
                        f_out.write(json.dumps(article, ensure_ascii=False) + '\n')
                        
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping malformed JSON line in Pass 2")

        print(f"--- Finished. Enriched data saved to: {output_filename} ---")
    
    except FileNotFoundError:
         print(f"ERROR: Input file not found at {input_path} during Pass 2.")
    except Exception as e:
        print(f"An error occurred during Pass 2: {e}")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    # Define all files to be processed and their metadata
    # Each tuple contains: (input_filename, output_filename, language_code, title_key, text_key)
    files_to_process = [
        ("wiki_en.jsonl", "wiki_en.jsonl", "en", "title_en", "text_en"),
        # # Monolingual files
        # ("wiki_zh_only.jsonl", "wiki_zh_only.jsonl", "zh", "title_zh", "text_zh"),
        # ("wiki_ja_only.jsonl", "wiki_ja_only.jsonl", "ja", "title_ja", "text_ja"),
        # ("wiki_ko_only.jsonl", "wiki_ko_only.jsonl", "ko", "title_ko", "text_ko"),
        # # Bilingual mapping files
        # ("wiki_zh_en_bilingual.jsonl", "wiki_zh_en_bilingual.jsonl", "zh", "title_zh", "text_zh"),
        # ("wiki_ja_en_bilingual.jsonl", "wiki_ja_en_bilingual.jsonl", "ja", "title_ja", "text_ja"),
        # ("wiki_ko_en_bilingual.jsonl", "wiki_ko_en_bilingual.jsonl", "ko", "title_ko", "text_ko"),
    ]

    # Loop through and process each file
    for input_file, output_file, lang, title_k, text_k in files_to_process:
        process_file(input_file, output_file, lang, title_k, text_k)

    print("\n--- All files have been processed successfully! ---")
