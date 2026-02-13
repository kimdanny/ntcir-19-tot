import json
import os
from tqdm import tqdm

# --- Configuration: List all the files you want to process ---
BASE_DIR = "YOUR_BASE_DIRECTORY_PATH"
INPUT_FILES = [
    "wikipedia_bilingual_zh_en.jsonl",
    "wikipedia_bilingual_ko_en.jsonl",
    "wikipedia_bilingual_ja_en.jsonl"
]


def clean_jsonl_file(input_path, output_path):
    """
    Reads a JSONL file, removes specified keys from each line, 
    and writes the cleaned data to a new file.
    (This function is unchanged)
    """
    # keys to remove for long entries
    keys_to_remove = ["url_en", "title_en", "text_en"]
    
    print(f"Starting to clean '{input_path}'...")
    print(f"Cleaned data will be saved to '{output_path}'")

    try:
        # Get the total number of lines for the tqdm progress bar
        with open(input_path, 'r', encoding='utf-8') as f_count:
            total_lines = sum(1 for line in f_count)

        with open(input_path, 'r', encoding='utf-8') as f_in, \
             open(output_path, 'w', encoding='utf-8') as f_out:

            for line in tqdm(f_in, total=total_lines, desc=f"Processing {os.path.basename(input_path)}"):
                # Python dict from JSON line
                data = json.loads(line)

                # Only need to clean if 'text_en' exists
                if 'text_en' in data:
                    for key in keys_to_remove:
                        if key in data:
                            del data[key]

                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

        print(f"\nCleaning complete for '{input_path}'!")
        print(f"Successfully created '{output_path}'")

    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_path}'")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    # --- Main Change: Loop through the list of files ---
    for filename in INPUT_FILES:
        # Construct the full input path for the current file
        input_path = os.path.join(BASE_DIR, filename)
        # Automatically generate the output filename
        output_path = input_path.replace(".jsonl", "_cleaned.jsonl")
        
        # Call the cleaning function for the current file
        clean_jsonl_file(input_path, output_path)
        print("-" * 60) # Add a separator for better readability between files
    
    print("All files have been processed.")
