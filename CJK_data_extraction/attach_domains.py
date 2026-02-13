import json
import os
import requests
import urllib.parse
from typing import Optional, Dict, List, Set
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
# --- NEW: Import libraries for retry logic ---
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ==============================================================================
# CONFIGURATION
# (This section is unchanged)
# ==============================================================================
BASE_INPUT_DIR = "YOUR_BASE_INPUT_DIRECTORY_PATH"
BASE_OUTPUT_DIR = "YOUR_BASE_OUTPUT_DIRECTORY_PATH"
FILES_TO_PROCESS = [
    {"input_file": "wiki_en.jsonl", "lang_code": "en", "title_key": "title_en"},
    # {"input_file": "wikipedia_zh_only.jsonl", "lang_code": "zh", "title_key": "title_zh"},
    # {"input_file": "wiki_zh_en_bilingual.jsonl", "lang_code": "zh", "title_key": "title_zh"},
    # {"input_file": "wiki_ja_only.jsonl", "lang_code": "ja", "title_key": "title_ja"},
    # {"input_file": "wiki_ja_en_bilingual.jsonl", "lang_code": "ja", "title_key": "title_ja"},
    # {"input_file": "wiki_ko_only.jsonl", "lang_code": "ko", "title_key": "title_ko"},
    # {"input_file": "wiki_ko_en_bilingual.jsonl", "lang_code": "ko", "title_key": "title_ko"}
]
BATCH_SIZE_URL = 30
BATCH_SIZE_QID = 50
MAX_WORKERS = 20
HEADERS = {'User-Agent': 'Wikidata Domain Enrichment Script (contact: YOUR_EMAIL@EMAIL.COM)'}

# ==============================================================================
# --- NEW: Function to create a session with automatic retries ---
# ==============================================================================
def build_session_with_retries():
    """Builds a requests.Session object with a robust retry strategy."""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,  # Total number of retries
        backoff_factor=1,  # Wait 1s, 2s, 4s between retries
        status_forcelist=[429, 500, 502, 503, 504],  # Retry on these HTTP status codes
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

# ==============================================================================
# CORE API FUNCTIONS (Now accept a 'session' object)
# ==============================================================================

def get_qids_for_titles_batch(session: requests.Session, titles: List[str], lang_code: str) -> Dict[str, str]:
    # ... (function logic is the same, but uses session.get) ...
    print(f"--- Step 1: Fetching Wikidata Q-IDs for {len(titles)} titles from {lang_code}.wikipedia... ---")
    title_to_qid_map = {}
    for i in tqdm(range(0, len(titles), BATCH_SIZE_URL), desc="Batch getting Q-IDs"):
        batch = titles[i:i + BATCH_SIZE_URL]
        encoded_titles = "|".join([urllib.parse.quote(str(title)) for title in batch])
        url = (
            f"https://{lang_code}.wikipedia.org/w/api.php?action=query&prop=pageprops"
            f"&ppprop=wikibase_item&titles={encoded_titles}&format=json&redirects"
        )
        try:
            response = session.get(url, headers=HEADERS, timeout=20) # CHANGE: Use session.get
            response.raise_for_status()
            data = response.json()
            redirect_map = {item['from']: item['to'] for item in data.get("query", {}, ).get("redirects", [])}
            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                api_title = page.get('title')
                qid = page.get("pageprops", {}).get("wikibase_item")
                if api_title and qid:
                    title_to_qid_map[api_title] = qid
            for original_title in batch:
                redirected_title = redirect_map.get(original_title)
                if redirected_title and redirected_title in title_to_qid_map:
                    title_to_qid_map[original_title] = title_to_qid_map[redirected_title]
        except Exception as e:
            print(f" -> API Error (get_qids): {e}")
    return title_to_qid_map

def fetch_instance_of_for_qid(session: requests.Session, qid: str) -> List[str]:
    # ... (function logic is the same, but uses session.get) ...
    params = {"action": "wbgetclaims", "entity": qid, "property": "P31", "format": "json"}
    try:
        response = session.get("https://www.wikidata.org/w/api.php", params=params, headers=HEADERS, timeout=10) # CHANGE: Use session.get
        if response.status_code == 200:
            data = response.json()
            claims = data.get("claims", {}).get("P31", [])
            return [
                claim["mainsnak"]["datavalue"]["value"]["id"]
                for claim in claims if "mainsnak" in claim and "datavalue" in claim["mainsnak"]
            ]
    except Exception:
        pass
    return []

def get_labels_for_qids_batch(session: requests.Session, qids: List[str]) -> Dict[str, str]:
    # ... (function logic is the same, but uses session.get) ...
    print("\n--- Step 3: Fetching English labels for all found domain Q-IDs... ---")
    qid_to_label_map = {}
    for i in tqdm(range(0, len(qids), BATCH_SIZE_QID), desc="Batch getting labels"):
        batch = qids[i:i + BATCH_SIZE_QID]
        params = {
            "action": "wbgetentities", "ids": "|".join(batch),
            "props": "labels", "languages": "en", "format": "json"
        }
        try:
            response = session.get("https://www.wikidata.org/w/api.php", params=params, headers=HEADERS, timeout=20) # CHANGE: Use session.get
            response.raise_for_status()
            data = response.json()
            entities = data.get("entities", {})
            for qid, entity_data in entities.items():
                label = entity_data.get("labels", {}).get("en", {}).get("value")
                if label:
                    qid_to_label_map[qid] = label
        except Exception as e:
            print(f"  -> API Error (get_labels): {e}")
    return qid_to_label_map

def process_single_file(input_file, output_file, lang_code, title_key):
    """Runs the entire enrichment process for a single input file."""
    
    if not os.path.exists(input_file):
        print(f"FATAL: Input file not found at '{input_file}'")
        return

    # --- NEW: Create a single session object with retry logic for this file's processing ---
    session = build_session_with_retries()
        
    with open(input_file, "r", encoding="utf-8") as f:
        articles = [json.loads(line) for line in f]
    
    unique_titles = sorted(list({article.get(title_key) for article in articles if article.get(title_key)}))
    print(f"Found {len(unique_titles)} unique titles to process from '{os.path.basename(input_file)}'.")
    
    # CHANGE: Pass the session to the API functions
    title_to_qid = get_qids_for_titles_batch(session, unique_titles, lang_code)
    print(f"Successfully mapped {len(title_to_qid)} titles to Wikidata Q-IDs.")

    qids_to_process = list(title_to_qid.values())
    qid_to_instance_of_qids = {}
    print("\n--- Step 2: Fetching 'instance of' (P31) properties in parallel... ---")
    # CHANGE: The ThreadPoolExecutor will now use our session with retries
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_qid = {executor.submit(fetch_instance_of_for_qid, session, qid): qid for qid in qids_to_process}
        for future in tqdm(as_completed(future_to_qid), total=len(qids_to_process), desc="Getting 'instance of'"):
            qid = future_to_qid[future]
            instance_of_list = future.result()
            if instance_of_list:
                qid_to_instance_of_qids[qid] = instance_of_list

    all_instance_of_qids = sorted(list({item for sublist in qid_to_instance_of_qids.values() for item in sublist}))
    # CHANGE: Pass the session to the API functions
    instance_of_qid_to_label = get_labels_for_qids_batch(session, all_instance_of_qids)
    print(f"Successfully fetched labels for {len(instance_of_qid_to_label)} unique domain Q-IDs.")

    # --- NEW CHANGE 1: Initialize a dictionary to count domain occurrences ---
    domain_counts = {}

    print("\n--- Final Step: Enriching articles and writing to output file... ---")
    with open(output_file, "w", encoding="utf-8") as f_out:
        for article in tqdm(articles, desc="Writing final data"):
            title = article.get(title_key)
            domains = []
            article_qid = title_to_qid.get(title)
            if article_qid:
                instance_of_qids = qid_to_instance_of_qids.get(article_qid, [])
                domains = [instance_of_qid_to_label.get(qid) for qid in instance_of_qids if instance_of_qid_to_label.get(qid)]
            article['domains'] = domains

            # --- NEW CHANGE 2: Update domain counts for statistics ---
            for domain in domains:
                domain_counts[domain] = domain_counts.get(domain, 0) + 1

            f_out.write(json.dumps(article, ensure_ascii=False) + '\n')

    print(f"\n--- Process Complete for {os.path.basename(input_file)}! ---")
    print(f"Enriched data has been saved to '{output_file}'.")

    # --- NEW CHANGE 3: Write the domain statistics to a separate file ---
    stats_output_file = output_file.replace(".jsonl", "_domain_stats.jsonl")
    print(f"\n--- Writing domain statistics to '{stats_output_file}'... ---")
    
    # Sort domains by count in descending order for readability
    sorted_domains = sorted(domain_counts.items(), key=lambda item: item[1], reverse=True)
    
    with open(stats_output_file, "w", encoding="utf-8") as f_stats:
        for domain, count in tqdm(sorted_domains, desc="Writing stats"):
            stat_record = {"domain": domain, "count": count}
            f_stats.write(json.dumps(stat_record, ensure_ascii=False) + '\n')
            
    print(f"Successfully wrote {len(sorted_domains)} unique domains with their counts.")

# ==============================================================================
# MAIN EXECUTION (Unchanged)
# ==============================================================================
if __name__ == "__main__":
    for file_config in FILES_TO_PROCESS:
        input_path = os.path.join(BASE_INPUT_DIR, file_config["input_file"])
        output_path = os.path.join(BASE_OUTPUT_DIR, file_config["input_file"])
        print(f"\n{'='*80}")
        print(f"STARTING PROCESSING FOR: {file_config['input_file']}")
        print(f"{'='*80}")
        process_single_file(
            input_file=input_path,
            output_file=output_path,
            lang_code=file_config["lang_code"],
            title_key=file_config["title_key"]
        )
    print("\n\n--- ALL FILES HAVE BEEN PROCESSED ---")