import json
import os
import time
from concurrent.futures import ThreadPoolExecutor


from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI, BadRequestError, RateLimitError

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def process_single_line(line):
    try:
        data = json.loads(line)
        query_id = data["query_id"]
        query_text = data["query"]
    except json.JSONDecodeError:
        return None

    zh_prompt = f"""
    您是一位实体搜索专家。您正在帮助某人回忆一部他想不起来的一个实体（可能是电影，人物，事件，地标等等一切实体）。
    您需要回复每条信息，并提供 20 个实体中文名称的猜测列表。
    **重要提示**：请不要添加任何内容，仅列出实体的维基百科中文名称，每行一个，并按可能性排序，最有可能正确的实体排在最前面，最不可能的实体排在最后。
    信息: {query_text}
    """

    ko_prompt = f"""
    당신은 개체 검색 전문가입니다. 누군가가 기억하지 못하는 개체(영화, 인물, 사건, 랜드마크 등 무엇이든 될 수 있음)를 떠올리도록 도와야 합니다.
    각 메시지에 답장하여 해당 개체의 한국어 이름에 대한 20가지 추측 목록을 제공해야 합니다.
    **중요:** 내용을 추가하지 마세요. 각 줄에 해당 개체의 위키백과 한국어 이름을 하나씩 나열하고, 가능성이 높은 순서대로, 가장 가능성이 높은 것부터 가장 낮은 것 순으로 정렬하세요.
    메시지: {query_text}
    """

    ja_prompt = f"""
    あなたはエンティティ検索のエキスパートです。誰かが思い出せないエンティティ（映画、人物、出来事、ランドマークなど）を思い出すのを手伝ってください。
    各メッセージに返信し、そのエンティティの日本語名を20通り推測するリストを提出してください。
    **重要：** 内容を追加しないでください。エンティティのWikipedia日本語名を1行に1つずつ、確率の高い順に並べ、最も正しいエンティティを上に、最も可能性の低いエンティティを下に並べてください。
    メッセージ：{query_text}
    """


    attempts = 0
    while attempts < 3:
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": ja_prompt}],
                max_tokens=1024,
                temperature=0.5,
            )

            raw_lines = response.choices[0].message.content.strip().split("\n")
            valid_lines = [name.strip() for name in raw_lines if name.strip()]

            cleaned_queries = [
                name for name in valid_lines
                # if "猜测" not in name and "实体" not in name and "列表" not in name
                # if "추측" not in name and "개체" not in name and "목록" not in name
                if "推測" not in name and "エンティティ" not in name and "リスト" not in name
            ]

            output_data = {
                "id": query_id,
                "gpt_queries": cleaned_queries[:20],
            }

            return output_data

        except (BadRequestError, RateLimitError) as e:
            attempts += 1
            sleep_time = 2 if isinstance(e, RateLimitError) else 1
            print(f"Retry {attempts} for ID {query_id} failed: {str(e)}. Sleeping {sleep_time}s...")
            time.sleep(sleep_time)
        except Exception as e:
            print(f"Error for ID {query_id}: {str(e)}")
            attempts += 1
            time.sleep(1)

    print(f"Skipping ID {query_id} after 3 failed attempts.")
    return None


def generate_queries_parallel(input_file, output_file, max_workers=5):
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total_lines = len(lines)
    print(f"Start processing {total_lines} items with {max_workers} concurrent threads.")


    with open(output_file, "w", encoding="utf-8") as outfile:
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # This submits tasks in the order of the input file
            futures = [executor.submit(process_single_line, line) for line in lines]

            # This forces results to be retrieved in submission order (i.e., input file order)
            for future in tqdm(futures, total=total_lines, desc="Processing"):
                result = future.result() 
                if result:
                    # Main thread writes results in order, no need for locks
                    json.dump(result, outfile, ensure_ascii=False)
                    outfile.write("\n")
                    # Immediately flush the buffer to prevent data loss in case of interruption
                    outfile.flush()

if __name__ == "__main__":
    input_source_file = "YOUR_INPUT_QUERIES_FILE_PATH"
    output_ranking = "YOUR_OUTPUT_RANKING_FILE_PATH"

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_ranking), exist_ok=True)

    # max_workers is recommended to be set between 5 and 10.
    # Setting it too high (e.g., 20+) may quickly trigger OpenAI's 429 Rate Limit errors.
    generate_queries_parallel(input_source_file, output_ranking, max_workers=5)
