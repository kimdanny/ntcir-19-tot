import csv
import os
# Removed threading/Lock imports as they are no longer needed for sequential writing
from concurrent.futures import ThreadPoolExecutor # Removed as_completed
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
import json
import logging
import time
from transformers import GPT2Tokenizer
from prompts import *

# Load OpenAI key from the .env file
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

REQUEST_TIMEOUT_SECONDS = 120
API_RETRY_ATTEMPTS = 3
API_RETRY_BACKOFF_SECONDS = 5
MAX_WORKERS = 8
BATCH_SIZE = 64
CHECKPOINT_ENABLED = True

# Configure prompt selection for each run.
DEFAULT_PROMPT_LANGUAGE = "en"
DATA_LANGUAGE = "ja"
PROMPT_LANGUAGE_FOR_ID_EN_PRESENT = "en"
PROMPT_LANGUAGE_FOR_ID_EN_MISSING = "en"

DEFAULT_TOT_PROMPT_TEMPLATE = TOT_PROMPTS_BY_LANGUAGE.get(
    DEFAULT_PROMPT_LANGUAGE, EN_TOT_PROMPT
)

TOT_SYSTEM_PROMPT_BY_TEMPLATE = {
    EN_TOT_PROMPT: TOT_SYSTEM_PROMPT_EN,
    EN_TOT_PROMPT_WITH_INST: TOT_SYSTEM_PROMPT_EN,
    KO_TRANS_TOT_PROMPT: TOT_SYSTEM_PROMPT_KO,
    ZH_TRANS_TOT_PROMPT: TOT_SYSTEM_PROMPT_ZH,
    JA_TRANS_TOT_PROMPT: TOT_SYSTEM_PROMPT_JA,
}


def get_tot_system_prompt(template):
    return TOT_SYSTEM_PROMPT_BY_TEMPLATE.get(template, TOT_SYSTEM_PROMPT_EN)


def call_chat_completion(messages, model, max_tokens, temperature):
    last_error = None
    for attempt in range(1, API_RETRY_ATTEMPTS + 1):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except Exception as e:
            last_error = e
            logging.warning(
                "OpenAI request failed (attempt %s/%s): %s",
                attempt,
                API_RETRY_ATTEMPTS,
                e,
            )
            if attempt < API_RETRY_ATTEMPTS:
                time.sleep(API_RETRY_BACKOFF_SECONDS * attempt)
    raise last_error


def get_checkpoint_path(output_file_path):
    return f"{output_file_path}.ckpt"


def load_checkpoint(checkpoint_path):
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return 0
    try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            value = f.read().strip()
        return int(value) if value else 0
    except Exception as e:
        logging.warning("Failed to read checkpoint %s: %s", checkpoint_path, e)
        return 0


def save_checkpoint(checkpoint_path, next_line_index):
    if not checkpoint_path:
        return
    temp_path = f"{checkpoint_path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        f.write(str(next_line_index))
    os.replace(temp_path, checkpoint_path)


def normalize_prompt_language(value):
    if value in TOT_PROMPTS_BY_LANGUAGE:
        return value
    return DEFAULT_PROMPT_LANGUAGE


def select_prompt_language(row_data):
    has_id_en = bool(row_data.get("id_en"))
    if has_id_en:
        return normalize_prompt_language(PROMPT_LANGUAGE_FOR_ID_EN_PRESENT)
    return normalize_prompt_language(PROMPT_LANGUAGE_FOR_ID_EN_MISSING)


def get_tot_prompt_template(prompt_language):
    if prompt_language == "en":
        if DATA_LANGUAGE != "en":
            return EN_TOT_PROMPT_WITH_INST
        return EN_TOT_PROMPT
    return TOT_PROMPTS_BY_LANGUAGE.get(prompt_language, DEFAULT_TOT_PROMPT_TEMPLATE)


def get_summarization_prompt_template(prompt_language):
    return SUMMARIZATION_USER_PROMPT_TEMPLATES_BY_LANGUAGE.get(
        prompt_language, SUMMARIZATION_USER_PROMPT_TEMPLATE_EN
    )


def get_summarization_system_prompt(prompt_language):
    return SUMMARIZATION_SYSTEM_PROMPTS_BY_LANGUAGE.get(
        prompt_language, SUMMARIZATION_SYSTEM_PROMPT_EN
    )


def build_summarization_prompt(input_text, prompt_language=DEFAULT_PROMPT_LANGUAGE):
    template = get_summarization_prompt_template(prompt_language)
    return template.format(input_text=input_text)


def build_summarization_messages(prompt, prompt_language=DEFAULT_PROMPT_LANGUAGE):
    system_prompt = get_summarization_system_prompt(prompt_language)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]


def split_document(document):
    """
    Splits the document (a long string) into paragraphs based on newlines.
    """
    # Split the document by newlines
    paragraphs = document.split("\n")
    # Filter out any empty strings that may occur due to consecutive newline characters
    return [para.strip() for para in paragraphs if para.strip()]


def extract_row_fields(row):
    return {
        "rel_doc_id": row.get("id"),
        "id_en": row.get("id_en"),
        "url": row.get("url"),
        "tgt_object": row.get("title"),
        "doc": row.get("text"),
        "domains_list": row.get("domains"),
    }


def determine_domain(domains_list):
    if not domains_list:
        return None
    if "film" in domains_list:
        return "film"
    if "human" in domains_list:
        return "human"
    return "general"


def generate_post_without_name(template, tgt_object, paragraphs):
    """
    Generates a post about the topic without mentioning its name, based on the template.
    """
    formatted_paragraphs = "\n".join(
        ["- " + para for para in paragraphs]
    )  # Format paragraphs for display
    prompt = template.format(
        ToTObject=tgt_object, Psg=formatted_paragraphs
    )

    system_prompt = get_tot_system_prompt(template)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    response = call_chat_completion(
        model="gpt-4o", messages=messages, max_tokens=1024, temperature=0.3
    )

    return response.choices[0].message.content.strip()


def summarize_paragraphs(paragraphs, prompt_language=DEFAULT_PROMPT_LANGUAGE):
    """
    Summarizes the given input paragraphs (a long string) into two paragraphs.
    """
    input_text = "\n\n".join(paragraphs if isinstance(paragraphs, list) else [paragraphs])
    
    prompt = build_summarization_prompt(truncated_text, prompt_language)
    messages = build_summarization_messages(prompt, prompt_language)

    response = call_chat_completion(
        model="gpt-4o", messages=messages, max_tokens=1024, temperature=0.5
    )

    return response.choices[0].message.content.strip()


def summarize_paragraphs_truncate_after_max_tokens(
    paragraphs, max_tokens=100000, prompt_language=DEFAULT_PROMPT_LANGUAGE
):
    """
    Summarizes the given input paragraphs into two paragraphs.
    Ensures that the input length in terms of tokens does not exceed the specified maximum.
    """
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")  # Load a GPT-2 tokenizer

    # Combine paragraphs into a single string
    input_text = "\n\n".join(paragraphs if isinstance(paragraphs, list) else [paragraphs])

    # Tokenize the input and check the token count
    tokens = tokenizer.encode(
        input_text, truncation=True, max_length=max_tokens, return_tensors="pt"
    )

    # Convert tokens back to text ensuring it does not exceed the max tokens
    truncated_text = tokenizer.decode(tokens[0], skip_special_tokens=True)

    prompt = build_summarization_prompt(input_text, prompt_language)
    messages = build_summarization_messages(prompt, prompt_language)

    response = call_chat_completion(
        model="gpt-4o", messages=messages, max_tokens=1024, temperature=0.5
    )

    return response.choices[0].message.content.strip()


def setup_logging(file_path):
    log_folder = file_path  # Specify the folder where you want to save the log files
    os.makedirs(log_folder, exist_ok=True)  # Ensure the log folder exists

    # Setup basic configuration for logging, adjust the filename to include the folder path
    logging.basicConfig(
        filename=os.path.join(log_folder, "process_log.txt"),
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Logger for tracking targets that didn't pass the check after 3 retries
    failed_logger = logging.getLogger("failed_objects")
    fh = logging.FileHandler(
        os.path.join(log_folder, "failed_objects_log.txt")
    )  # Adjust the filename to include the folder path
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)
    failed_logger.addHandler(fh)
    return failed_logger


# --- Added: Worker function to process a single line and RETURN the result ---
def process_single_row_worker(line, failed_logger, i, input_file_path):
    """
    Worker function to process a single JSON line. 
    Instead of writing to file directly, it RETURNS the processed data.
    This ensures we can write them in strict order in the main thread.
    """
    try:
        row = json.loads(line)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON on line {i+1} in {input_file_path}: {e}")
        return None

    retries = 0
    
    row_data = extract_row_fields(row)
    rel_doc_id = row_data["rel_doc_id"]
    tgt_object = row_data["tgt_object"]
    doc = row_data["doc"]
    domains_list = row_data["domains_list"]

    logging.info("Start line %s (ID: %s)", i + 1, rel_doc_id)

    if not all([rel_doc_id, tgt_object, doc]):
        logging.info("Skip line %s due to missing fields (ID: %s)", i + 1, rel_doc_id)
        return None
    
    domain = determine_domain(domains_list)
    if domain is None:
        logging.info("Skip line %s due to missing domains (ID: %s)", i + 1, rel_doc_id)
        return None
    topic = domains_list 

    try:
        prompt_language = select_prompt_language(row_data)
        tot_prompt_template = get_tot_prompt_template(prompt_language)
        summarization_prompt_template = get_summarization_prompt_template(prompt_language)
        summarization_system_prompt = get_summarization_system_prompt(prompt_language)
        tot_system_prompt = get_tot_system_prompt(tot_prompt_template)
        print(
            f"[prompts] id_en={bool(row_data.get('id_en'))} language={prompt_language}\n"
            f"[tot_system]\n{tot_system_prompt}\n"
            f"[tot_user]\n{tot_prompt_template}\n"
            f"[summarization_system]\n{summarization_system_prompt}\n"
            f"[summarization_user]\n{summarization_prompt_template}\n"
        )

        # 1. Summarize
        try:
            summarization = summarize_paragraphs(doc, prompt_language=prompt_language)
        except Exception as e:
            summarization = summarize_paragraphs_truncate_after_max_tokens(
                doc, prompt_language=prompt_language
            )
        
        # 2. Split
        paragraphs = split_document(summarization)

        if not paragraphs or not any(paragraphs): 
            failed_logger.error(f"ID: {rel_doc_id}, Title: {tgt_object} has empty paragraphs.")
            return None

        # 3. Generate (Retries)
        processed_response = None
        while retries < 3:
            try:
                response = generate_post_without_name(tot_prompt_template, tgt_object, paragraphs)
            except Exception as e:
                failed_logger.error(f"GPT API error for ID: {rel_doc_id}: {e}")
                break 

            if not response: 
                failed_logger.error(f"Empty response for ID: {rel_doc_id}")
                break 

            if tgt_object.lower() in (response or "").lower():
                retries += 1
                if retries >= 3:
                    failed_logger.error(f"ID: {rel_doc_id} failed to generate valid response after 3 retries.")
                    break
                continue 

            processed_response = response.replace("\n", " ").strip('"')
            break

        # --- Return the result instead of writing ---
        if processed_response:
            logging.info("Finished line %s (ID: %s)", i + 1, rel_doc_id)
            return {
                "status": "success",
                "domain": domain,
                "rel_doc_id": rel_doc_id,
                "rel_doc_title": tgt_object,
                "query": processed_response,
                "debug_info": {
                    "topic": topic,
                    "ToTObject": tgt_object,
                    "paragraphs": paragraphs,
                    "prompt": tot_prompt_template,
                    "response": response,
                    "rel_doc_id": rel_doc_id
                }
            }
        else:
            logging.info("Finished line %s without response (ID: %s)", i + 1, rel_doc_id)
            return None

    except Exception as e:
        failed_logger.error(f"Failed to process ID: {rel_doc_id}, Title: {tgt_object} error: {e}")
        logging.info("Finished line %s with error (ID: %s)", i + 1, rel_doc_id)
        return None


def process_single_file(
    input_file_path, output_file_path, json_folder, failed_logger, query_id_counter
):
    """
    Modified to use ThreadPoolExecutor for concurrency but preserve strict order.
    """
    
    try:
        # --- Added: Read all lines first ---
        with open(input_file_path, "r", encoding="utf-8") as infile:
            lines = infile.readlines()

        total_lines = len(lines)
        checkpoint_path = get_checkpoint_path(output_file_path) if CHECKPOINT_ENABLED else None
        start_line_index = load_checkpoint(checkpoint_path) if CHECKPOINT_ENABLED else 0

        if start_line_index >= total_lines:
            logging.info(
                "Checkpoint indicates completion for %s (total lines: %s).",
                input_file_path,
                total_lines,
            )
            return query_id_counter

        lines_to_process = lines[start_line_index:]
        total_to_process = len(lines_to_process)

        output_mode = "w"
        if start_line_index > 0 and os.path.exists(output_file_path):
            output_mode = "a"
            try:
                with open(output_file_path, "r", encoding="utf-8") as existing_file:
                    existing_output_lines = sum(1 for _ in existing_file)
                if query_id_counter <= 1:
                    query_id_counter = existing_output_lines + 1
            except Exception as e:
                logging.warning("Failed to read existing output %s: %s", output_file_path, e)
        elif start_line_index > 0:
            logging.warning(
                "Checkpoint found but output missing; restarting output for %s.",
                output_file_path,
            )

        # --- Process in batches to avoid long in-flight queues ---
        with open(output_file_path, output_mode, encoding="utf-8") as outfile:
            with tqdm(
                total=total_to_process,
                desc=f"Processing {os.path.basename(input_file_path)}",
                unit="doc",
            ) as pbar:
                for batch_start in range(0, total_to_process, BATCH_SIZE):
                    batch_lines = lines_to_process[batch_start: batch_start + BATCH_SIZE]
                    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                        futures = []
                        for offset, line in enumerate(batch_lines):
                            line_index = start_line_index + batch_start + offset
                            futures.append(
                                executor.submit(
                                    process_single_row_worker,
                                    line,
                                    failed_logger,
                                    line_index,
                                    input_file_path,
                                )
                            )

                        for offset, future in enumerate(futures):
                            line_index = start_line_index + batch_start + offset
                            try:
                                result = future.result()
                            except Exception as e:
                                logging.error(
                                    "Worker failed at line %s in %s: %s",
                                    line_index + 1,
                                    input_file_path,
                                    e,
                                )
                                result = None

                            if result and result["status"] == "success":
                                # Write Main JSONL
                                query_id_str = f"{query_id_counter:04d}"
                                output_row = {
                                    "query_id": query_id_str,
                                    "source": "synthetic",
                                    "llm": "gpt-4o",
                                    "domain": result["domain"],
                                    "rel_doc_id": result["rel_doc_id"],
                                    "rel_doc_title": result["rel_doc_title"],
                                    "query": result["query"],
                                }
                                outfile.write(json.dumps(output_row, ensure_ascii=False) + "\n")
                                outfile.flush()

                                query_id_counter += 1

                                # Write Debug JSON
                                debug_info = result["debug_info"]
                                tgt_object = debug_info["ToTObject"]
                                rel_doc_id = debug_info["rel_doc_id"]
                                # Remove extra keys not needed for debug file
                                debug_output = {k: v for k, v in debug_info.items() if k != "rel_doc_id"}

                                safe_title = "".join(
                                    c for c in tgt_object if c.isalnum() or c in (" ", "_")
                                ).rstrip()
                                file_name = os.path.join(
                                    json_folder,
                                    f"{rel_doc_id}_{safe_title.replace(' ', '_')}.json",
                                )
                                with open(file_name, "w", encoding="utf-8") as f:
                                    json.dump(debug_output, f, indent=4, ensure_ascii=False)

                            pbar.update(1)
                            if CHECKPOINT_ENABLED:
                                save_checkpoint(checkpoint_path, line_index + 1)

        if CHECKPOINT_ENABLED and checkpoint_path:
            try:
                if start_line_index + total_to_process >= total_lines:
                    os.remove(checkpoint_path)
            except OSError:
                pass

    except FileNotFoundError:
        logging.error(f"Input file not found: {input_file_path}")
    except Exception as e:
        logging.error(f"An unexpected error occurred while processing {input_file_path}: {e}")

    logging.info(f"Finished processing {input_file_path}. Total queries generated so far: {query_id_counter}")
    
    return query_id_counter


def main_processing():
    """
    Main function to orchestrate the processing of multiple files with a continuous query_id.
    """
    # --- Configuration ---
    # Directory containing 'train.jsonl', 'test.jsonl', 'validation.jsonl'
    input_dir = "YOUR_INPUT_DIR_PATH"
    # Directory where the new 'train.jsonl', 'test.jsonl', 'validation.jsonl' will be saved
    output_dir = "YOUR_OUTPUT_DIR_PATH"
    # Directory for optional debug JSON files
    json_folder = "YOUR_DEBUG_JSON_OUTPUT_DIR"
    # Directory for log files
    log_folder_path = "YOUR_LOG_DIR_PATH"

    # Files to process in order to maintain continuous query_id
    files_to_process = ["YOUR_INPUT_FILENAME.jsonl"]
    # --- End Configuration ---

    # --- Setup (Done once) ---
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(json_folder, exist_ok=True)
    failed_logger = setup_logging(log_folder_path)
    
    query_id_counter = 1 # Initialize the continuous counter
    # --- End Setup ---

    logging.info(f"Starting batch processing. Input: {input_dir}, Output: {output_dir}")

    for file_name in files_to_process:
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)

        # Check if the input file exists before trying to process
        if not os.path.exists(input_path):
            logging.warning(f"Input file not found, skipping: {input_path}")
            continue

        logging.info(f"--- Processing file: {input_path} ---")
        
        # Call the processing function, passing and receiving the updated counter
        query_id_counter = process_single_file(
            input_file_path=input_path,
            output_file_path=output_path,
            json_folder=json_folder,
            failed_logger=failed_logger,
            query_id_counter=query_id_counter
        )
        
        logging.info(f"--- Finished file: {input_path}. Next query ID will be: {query_id_counter} ---")
    
    logging.info(f"All files processed successfully. Total queries generated: {query_id_counter}")


if __name__ == "__main__":
    # Call the main processing function
    main_processing()
