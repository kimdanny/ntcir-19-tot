import csv
import os
from typing import Dict, List
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, pipeline

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- Model ID (loaded from a local path) ---
MODEL_ID = "YOUR_TRANSLATION_MODEL_PATH"

# --- Base directory for input and output files ---
INPUT_DIR = "YOUR_INPUT_DIRECTORY_PATH"
ARTIFACTS_DIR = "YOUR_OUTPUT_DIRECTORY_PATH"

# --- New Change: Added a batch size for the translation pipeline ---
# This controls how many items are processed in parallel on the GPU.
# Adjust based on your GPU's VRAM. 32 is a reasonable start for an A100.
BATCH_SIZE = 16

# --- Target language configuration ---
# The keys are the language codes used in filenames and new columns.
# The values are the full language names required by the translation model's prompt.
TARGET_LANGUAGES = {
    "zh": "Chinese (Simplified)",
    "ja": "Japanese",
    "ko": "Korean"
}

# ==============================================================================
# CORE FUNCTIONS
# ==============================================================================

class Translator:
    """A wrapper class for the translation model to load it only once."""
    def __init__(self, model_id: str):
        print(f"--- Loading Translation Model: {model_id} ---")
        print("This may take a few minutes. Requires a GPU (e.g., A100).")
        
        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch.bfloat16, # Use bfloat16 for better performance on modern GPUs
            device_map="auto",
        )
        self.tokenizer = self.pipe.tokenizer
        print("--- Model loaded successfully! ---")

    # --- New Change: Replaced the single-translation method with a batch method ---
    def translate_batch(self, prompts: List[str]) -> List[str]:
        """Translates a batch of prompts at once for maximum efficiency."""
        
        # Prepare all prompts for the pipeline
        all_messages = [[{"role": "user", "content": prompt}] for prompt in prompts]

        try:
            # Pass the entire list to the pipeline for parallel processing on the GPU
            print(f"  > Translating a batch of {len(prompts)} items...")
            outputs = self.pipe(
                all_messages,
                max_new_tokens=4096,
                do_sample=False,
                batch_size=BATCH_SIZE # Inform the pipeline of our batch size
            )
            
            # Extract the results from each output
            results = []
            for output in outputs:
                assistant_reply = output[0]["generated_text"][-1]
                results.append(assistant_reply['content'].strip())
            print("  > Batch translation complete.")
            return results

        except Exception as e:
            print(f"    -> Batch Translation Error: {e}")
            return [f"[TRANSLATION FAILED: {e}]"] * len(prompts)

def process_file_for_translation(translator: Translator, lang_code: str, lang_name: str):
    """
    Reads an enriched TSV file, translates the 'query' column for each row in a batch,
    and writes the result to a new file with an added '{lang}_query' column.
    """
    input_filename = f"{lang_code}_output.tsv"
    output_filename = f"{lang_code}-test-2025-mapping.tsv"

    input_path = os.path.join(INPUT_DIR, input_filename)
    output_path = os.path.join(ARTIFACTS_DIR, output_filename)

    if not os.path.exists(input_path):
        print(f"--- SKIPPING: Input file not found for '{lang_name}': {input_path} ---")
        return

    print(f"\n--- Starting processing for {lang_name} ---")
    print(f"Reading from: {input_path}")
    print(f"Writing to:   {output_path}")

    # --- New Change: The logic is now a 3-step batch process ---

    # Step 1: Read all rows and collect all queries to be translated.
    with open(input_path, 'r', encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in, delimiter='\t')
        rows = list(reader)
        if not rows: # Handle empty file case
            print(f"--- SKIPPING: Input file for '{lang_name}' is empty. ---")
            return
        original_headers = reader.fieldnames
    
    prompts_to_translate = []
    for row in rows:
        english_query = row['query']
        prompt = (
            f"Translate the following English source text to {lang_name}:\n"
            f"English: {english_query}\n"
            f"{lang_name}: "
        )
        prompts_to_translate.append(prompt)

    # Step 2: Perform all translations in a single, efficient batch call.
    translated_queries = translator.translate_batch(prompts_to_translate)
    
    # Step 3: Write the results to the new file.
    new_query_column = f"{lang_code}_query"
    new_headers = original_headers + [new_query_column]

    with open(output_path, 'w', encoding='utf-8', newline='') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=new_headers, delimiter='\t')
        writer.writeheader()

        # Iterate through original rows and the corresponding translations
        for i, row in enumerate(tqdm(rows, desc=f"Writing results for {lang_name}")):
            # Add the new translated column to the row
            row[new_query_column] = translated_queries[i]
            # Write the completed row to the new file
            writer.writerow(row)
            
    print(f"--- Finished processing for {lang_name}. Output saved. ---")


def main():
    """Main processing loop."""
    
    # 1. Initialize the translator once for all files (this is the slow, one-time setup)
    translator = Translator(MODEL_ID)

    # 2. Loop through each language and process its corresponding file
    for lang_code, lang_name in TARGET_LANGUAGES.items():
        process_file_for_translation(translator, lang_code, lang_name)
    
    print("\n--- All tasks complete! ---")
    for lang_code in TARGET_LANGUAGES.keys():
        print(f"Final translated file created at: {os.path.join(ARTIFACTS_DIR, f'{lang_code}-test-2025-mapping.tsv')}")


if __name__ == "__main__":
    main()

