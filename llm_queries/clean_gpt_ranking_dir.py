import json
import os

def clean_entity_names(input_file, output_file):
    """
    Processes a single file. (Logic remains the same as your original code)
    """
    with open(input_file, "r", encoding="utf-8") as infile, open(
        output_file, "w", encoding="utf-8"
    ) as outfile:
        for line in infile:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue # Skip invalid lines if any

            cleaned_movies = []
            CHARS_TO_STRIP = ' \n\t"《》'

            if "gpt_queries" in data:
                for movie in data["gpt_queries"]:
                    # 1. Remove order number (e.g., "1. ")
                    cleaned_name = (
                        ". ".join(movie.split(". ")[1:]) if ". " in movie else movie
                    )

                    # 2. Remove dash/bullet prefix (e.g., "- ")
                    if cleaned_name.startswith("- "):
                        cleaned_name = cleaned_name[2:]
                    
                    # 3. Remove all surrounding junk chars
                    cleaned_name = cleaned_name.strip(CHARS_TO_STRIP)

                    # 4. Only add if the name is not empty
                    if cleaned_name:
                        cleaned_movies.append(cleaned_name)

                data["gpt_queries"] = cleaned_movies

            json.dump(data, outfile, ensure_ascii=False)
            outfile.write("\n")

def process_directory(input_dir, output_dir):
    # [NEW] Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Iterate over all files in the input directory
    for filename in os.listdir(input_dir):
        # Filter to process only .jsonl files
        if filename.endswith(".jsonl"):
            # Construct full file paths
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            print(f"Processing: {filename} ...")
            clean_entity_names(input_path, output_path)

    print("All files processed.")

if __name__ == "__main__":
    # Define directory paths instead of single file paths
    input_directory = "YOUR_INPUT_DIRECTORY_PATH"
    
    # You can save to a new folder to keep things organized
    output_directory = "YOUR_OUTPUT_DIRECTORY_PATH"

    process_directory(input_directory, output_directory)