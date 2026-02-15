# NTCIR ToT (Multilingual Retrieval)

This repository contains scripts for:

- Building CJK Wikipedia-aligned corpora (English/Chinese/Japanese/Korean).
- Generating LLM-based Tip-of-the-Tongue (ToT) queries.
- Running lexicon and dense retrieval baselines for multilingual ToT evaluation.

The codebase is script-driven (no single orchestrator). Most extraction scripts require editing constants in-file (`YOUR_*` placeholders).

## Repository Layout

- `CJK_data_extraction/`: dataset creation/enrichment scripts (Wikipedia alignment, domain labels, popularity/doc length, CJK mapping, translations).
- `llm_queries/`: LLM query generation and GPT ranking helper scripts.
- `model_inference/`: retrieval/evaluation pipelines (`run_lexicon_*`, `run_dense_*`, `gpt_post_*`, `tot_*` dataset registration).
- `CJK_data_stats/`: distribution/correlation analysis scripts.
- `analysis/`: notebook for system-rank correlation (`tau_correlation.ipynb`).
- `output/`: local logs/debug outputs.

## Environment Setup

This repo targets Python 3.9.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For OpenAI-backed scripts, create a local `.env` file in the repo root and set:

```bash
OPENAI_API_KEY=your_api_key_here
```

## Expected Dataset Format for Inference

`model_inference/tot_en.py`, `model_inference/tot_zh.py`, `model_inference/tot_ja.py`, `model_inference/tot_ko.py` register local datasets via `ir_datasets`.

Expected layout:

```text
<DATA_PATH>/
  corpus.jsonl
  train/
    queries.jsonl
    qrel.txt
  dev/
    queries.jsonl
    qrel.txt
  test/
    queries.jsonl
    qrel.txt   # optional
```

Expected key fields:

- `corpus.jsonl` records: `id`, `url`, `title`, `text`, `domains`, `id_en`, `popularity`, `doc_length`
- `queries.jsonl` records: `query_id`, `source`, `llm`, `domain`, `rel_doc_id`, `rel_doc_title`, `query`

## Workflow 1: Build/Enrich CJK Corpora

Most scripts below use in-file constants such as `YOUR_OUTPUT_DIRECTORY_PATH`, `YOUR_DATASET_PATH_HERE`, `/dataset/`, or `/artifacts/`.
Update those values before running.

1. Export English corpus:

```bash
python CJK_data_extraction/wikidata_en.py
```

2. Create language corpora and EN-linked bilingual files:

```bash
python CJK_data_extraction/wikidata_zh.py
python CJK_data_extraction/wikidata_ja.py
python CJK_data_extraction/wikidata_ko.py
```

3. Optional cleanup/lookup utilities:

```bash
python CJK_data_extraction/wikidata_cleaning.py
python CJK_data_extraction/build_english_look_up.py
python CJK_data_extraction/en_wiki_load.py
```

4. Add metadata:

```bash
python CJK_data_extraction/attach_domains.py
python CJK_data_extraction/attach_popularity_doclen.py
```

## Workflow 2: Generate LLM ToT Queries

Main script:

```bash
python llm_queries/generate_gpt_queries.py
```

Before running, edit configuration constants in `main_processing()`:

- `input_dir`
- `output_dir`
- `json_folder`
- `log_folder_path`
- `files_to_process`

The input JSONL rows are expected to include fields like `id`, `id_en`, `title`, `text`, and `domains`.

Output includes:

- query JSONL files with fields such as `query_id`, `domain`, `rel_doc_id`, `rel_doc_title`, `query`
- per-entity debug JSON files
- logs in `process_log.txt` / `failed_objects_log.txt`

## Workflow 3: Retrieval & Evaluation

### Lexicon Retrieval

Use language-specific scripts:

- `model_inference/run_lexicon_en.py`
- `model_inference/run_lexicon_zh.py`
- `model_inference/run_lexicon_ja.py`
- `model_inference/run_lexicon_ko.py`

Example:

```bash
python model_inference/run_lexicon_en.py \
  --data_path /path/to/dataset \
  --split test \
  --field text \
  --query text \
  --index_name en_qld_test \
  --param_mu 1000 \
  --run runs/en_qld_test.run \
  --run_format trec_eval \
  --run_id en_qld_test
```

Note: current `run_lexicon_*` code uses `searcher.set_qld(...)` by default (LMDirichlet). BM25 lines are present but commented.

### Dense Retrieval

Use:

- `model_inference/run_dense_en.py`
- `model_inference/run_dense_zh.py`
- `model_inference/run_dense_ja.py`
- `model_inference/run_dense_ko.py`

Example:

```bash
python model_inference/run_dense_en.py \
  --data_path /path/to/dataset \
  --model_or_checkpoint sentence-transformers/all-mpnet-base-v2 \
  --embed_size 768 \
  --encode_batch_size 128 \
  --query text \
  --model_dir results/dense_en \
  --run_id dense_en_test \
  --device cuda
```

This pipeline currently performs encoding/retrieval/evaluation with pretrained checkpoints; training blocks are mostly disabled in code.

### GPT Ranking Post-processing to Run Files

1. Generate ranked guesses from GPT:

Before running, edit these placeholders in `llm_queries/rank_by_gpt.py`:

- `input_source_file`
- `output_ranking`

```bash
python llm_queries/rank_by_gpt.py
```

2. Clean ranking text:

Before running, set placeholder paths in one of the following scripts:

- `llm_queries/clean_gpt_ranking.py`: `input_raw_ranking_file`, `output_cleaned_ranking_file`
- `llm_queries/clean_gpt_ranking_dir.py`: `input_directory`, `output_directory`

```bash
python llm_queries/clean_gpt_ranking.py
# or
python llm_queries/clean_gpt_ranking_dir.py
```

3. Convert cleaned rankings to TREC run files:

```bash
python model_inference/gpt_post_en.py \
  --input /path/to/cleaned.jsonl \
  --split test \
  --data_path /path/to/dataset \
  --index_name gpt_title_index_en \
  --run runs/gpt_en.run \
  --run_format trec_eval \
  --run_id gpt_en
```

Equivalent scripts exist for `zh`, `ja`, and `ko`.

## Analysis Utilities

- `CJK_data_stats/CJK_data_visual.py`
- `CJK_data_stats/CJK_pop_doclen_correlation.py`
- `CJK_data_stats/CJK_pop_doclen_stratified.py`
- `analysis/tau_correlation.ipynb`

These scripts also use in-file path constants and should be edited before execution.

## Practical Notes

- Many scripts have `YOUR_*` placeholders and are not turnkey until configured.
- Chinese pipelines use OpenCC normalization in several inference scripts.
- `pyserini`-based scripts require Java and Lucene-compatible setup.
- Keep `.env`, `output/logs/`, and `output/json_output/` local.
