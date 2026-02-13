import ir_datasets
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses, models
from src import data, encode, utils
import pytrec_eval
from torch import nn
import argparse
import os
import logging
from collections import defaultdict

import tot_ja
import run_lexicon_ja

from FlagEmbedding import BGEM3FlagModel
import scipy.sparse as sp
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, AutoModel, DPRContextEncoder, DPRContextEncoderTokenizer
import torch
import numpy as np
from tqdm import tqdm

import pytorch_lightning as pl
import torch.nn.init as init

log = logging.getLogger(__name__)


class CustomDPRModel:
    def __init__(self, model_name, device="cuda"):
        super().__init__()
        self.tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(model_name)
        self.model = DPRQuestionEncoder.from_pretrained(model_name)
        self.device = device
        self.model.to(self.device)

    def encode(
        self,
        texts,
        batch_size=256,
        convert_to_numpy=True,
        show_progress_bar=False,
        device="cuda",
    ):
        all_embeddings = []
        iterable = tqdm(
            range(0, len(texts), batch_size),
            desc="Encoding",
            disable=not show_progress_bar,
        )
        for i in iterable:
            batch_texts = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch_texts, return_tensors="pt", padding=True, truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs).pooler_output
            if convert_to_numpy:
                outputs = outputs.cpu().numpy()
            all_embeddings.append(outputs)
        if convert_to_numpy:
            return np.concatenate(all_embeddings, axis=0)
        return torch.cat(all_embeddings, dim=0)

    def eval(self):
        self.model.eval()


class LouisDPRModel(pl.LightningModule):
    def __init__(self, model_name):
        super().__init__()
        self.lm = AutoModel.from_pretrained(model_name)
        self.projection = nn.Linear(1024, 1024)
        self.tanh = nn.Tanh()

    def encode(self, inputs, attention_mask):
        model_output = self.lm(inputs, attention_mask)
        token_embeddings = model_output[0]
        cls_embeddings = self.cls_pooling(token_embeddings)
        return cls_embeddings

    def mean_pooling(self, x, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(x.size()).float()
        return torch.sum(x * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def cls_pooling(self, x):
        return x[:, 0]

    def forward(self, input_ids, attention_mask=None):
        cls_embeddings = self.encode(input_ids, attention_mask)
        projected = self.projection(cls_embeddings)
        activated = self.tanh(projected)
        return activated

    def load_model(self, model_path):
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            # Remove the 'lm.' prefix and load
            new_state_dict = {
                key.replace("lm.", ""): value for key, value in state_dict.items()
            }
            # Separate state dict for the pre-trained and custom layers
            lm_state_dict = {
                key: val
                for key, val in new_state_dict.items()
                if not key.startswith("projection")
            }
            projection_state_dict = {
                key: val
                for key, val in new_state_dict.items()
                if key.startswith("projection")
            }

            self.lm.load_state_dict(lm_state_dict, strict=False)  # Load the model part
            self.projection.load_state_dict(
                projection_state_dict, strict=False
            )  # Load the projection part
            print("Model loaded successfully.")
        except RuntimeError as e:
            print("Failed to load model state dictionary!")
            print(e)

    def eval_model(self):
        self.eval()


class MultilingualDPRWrapper:
    """
    For voidful/dpr and standard HuggingFace DPR dual encoder models.
    Automatically loads paired Question Encoder and Context Encoder.
    """
    def __init__(self, model_name, device="cuda"):
        self.device = device
        self.is_query_mode = True # DEFAULT to query mode
        
        print(f"Loading DPR Question Encoder: {model_name}...")
        self.q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(model_name)
        self.q_model = DPRQuestionEncoder.from_pretrained(model_name).to(self.device)
        
        # Automatically derive Context Encoder name
        if "question_encoder" in model_name:
            ctx_model_name = model_name.replace("question_encoder", "ctx_encoder")
        else:
            ctx_model_name = model_name
            
        print(f"Loading DPR Context Encoder: {ctx_model_name}...")
        try:
            self.ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(ctx_model_name)
            self.ctx_model = DPRContextEncoder.from_pretrained(ctx_model_name).to(self.device)
        except OSError:
            print(f"Warning: Could not find Context Encoder at {ctx_model_name}. Using Question Encoder shared weights.")
            self.ctx_tokenizer = self.q_tokenizer
            self.ctx_model = self.q_model

    def set_query_mode(self, is_query: bool):
        self.is_query_mode = is_query

    def eval(self):
        self.q_model.eval()
        self.ctx_model.eval()

    def encode(
        self,
        texts,
        batch_size=256,
        convert_to_numpy=True,
        show_progress_bar=False,
        device="cuda",
        **kwargs
    ):
        if self.is_query_mode:
            tokenizer = self.q_tokenizer
            model = self.q_model
            desc = "Encoding Queries"
        else:
            tokenizer = self.ctx_tokenizer
            model = self.ctx_model
            desc = "Encoding Docs"

        all_embeddings = []
        if isinstance(texts, str):
            texts = [texts]
            
        iterable = tqdm(
            range(0, len(texts), batch_size),
            desc=desc,
            disable=not show_progress_bar,
        )
        
        model.eval()
        
        for i in iterable:
            batch_texts = texts[i : i + batch_size]
            
            inputs = tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512 # DPR 通常支持 512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs).pooler_output
                
            if convert_to_numpy:
                outputs = outputs.cpu().numpy()
            all_embeddings.append(outputs)
            
        if convert_to_numpy:
            return np.concatenate(all_embeddings, axis=0)
        return torch.cat(all_embeddings, dim=0)


class DPRXMWrapper:
    """
    DPR-XM (XMOD Backbone) Wrapper.
    """
    def __init__(self, model_name, device="cuda"):
        print(f"Loading DPR-XM Model: {model_name}...")
        
        self.model = SentenceTransformer(model_name, device=device)
        
        language_code = "ja_XX" 
        try:
            self.model[0].auto_model.set_default_language(language_code)
            print(f"DPR-XM Language Adapter set to: {language_code}")
        except AttributeError:
            print("Warning: Could not set language adapter. Is this really an XMOD model?")

        # FP16 for efficiency
        self.model.half()
        
        self.is_query_mode = False 

    def set_query_mode(self, is_query: bool):
        self.is_query_mode = is_query

    def eval(self):
        self.model.eval()

    def encode(
        self,
        texts,
        batch_size=256,
        convert_to_numpy=True,
        show_progress_bar=True,
        **kwargs
    ):
        if isinstance(texts, str):
            texts = [texts]

        return self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=convert_to_numpy,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=True 
        )


class MContrieverWrapper:
    """
    Wrapper for Facebook mContriever.
    Architecture: BERT-based with Mean Pooling.
    Precision: Float32 (Standard).
    """
    def __init__(self, model_name, device="cuda"):
        print(f"Loading mContriever: {model_name}...")
        from transformers import AutoTokenizer, AutoModel
        
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        
        # mContriever 不需要 query 指令
        self.is_query_mode = False 

    def set_query_mode(self, is_query: bool):
        self.is_query_mode = is_query

    def eval(self):
        self.model.eval()

    def _mean_pooling(self, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1)
        sentence_embeddings = sentence_embeddings / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    def encode(
        self,
        texts,
        batch_size=256,
        convert_to_numpy=True,
        show_progress_bar=True,
        **kwargs
    ):
        if isinstance(texts, str):
            texts = [texts]
            
        all_embeddings = []
        iterable = tqdm(
            range(0, len(texts), batch_size),
            desc="Encoding mContriever",
            disable=not show_progress_bar,
        )
        
        for i in iterable:
            batch_texts = texts[i : i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors='pt'
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)

                token_embeddings = outputs.last_hidden_state
                embeddings = self._mean_pooling(token_embeddings, inputs['attention_mask'])
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
            if convert_to_numpy:
                embeddings = embeddings.cpu().numpy()
            all_embeddings.append(embeddings)
            
        if convert_to_numpy:
            return np.concatenate(all_embeddings, axis=0)
        return torch.cat(all_embeddings, dim=0)


class BGEM3Wrapper:
    """BGE-M3 Dense Mode (1024 dim)"""
    def __init__(self, model_name, device="cuda"):
        print(f"Loading BGE-M3 Model (Dense): {model_name}...")
        
        self.model = BGEM3FlagModel(model_name, use_fp16=True, device=device)
        self.is_query_mode = False 

    def set_query_mode(self, is_query: bool):
        self.is_query_mode = is_query

    def eval(self):
        pass

    def encode(self, texts, batch_size=32, convert_to_numpy=True, show_progress_bar=True, **kwargs):
        if isinstance(texts, str): texts = [texts]

        embeddings_dict = self.model.encode(
            texts, 
            batch_size=batch_size, 
            max_length=1024, 
            return_dense=True, 
            return_sparse=False, 
            return_colbert_vecs=False
        )
        embeddings = embeddings_dict['dense_vecs']
        
        if convert_to_numpy and torch.is_tensor(embeddings):
            return embeddings.cpu().detach().numpy()
        return embeddings


class BGEM3SparseWrapper:
    """
    BGE-M3 Sparse Mode (250002 dim) - Device Sync Fix
    This wrapper ensures that the internal modules are properly moved to the specified device,
    addressing potential issues with device mismatches during encoding.
    """
    def __init__(self, model_name, device="cuda"):
        print(f"Loading BGE-M3 (Sparse Mode - Device Sync): {model_name}...")
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError:
            raise ImportError("Please run: pip install -U FlagEmbedding")
        
        self.device = device
        self.flag_model = BGEM3FlagModel(model_name, use_fp16=True, device=device)
        self.tokenizer = self.flag_model.tokenizer
        self.vocab_size = 250002
        self.is_query_mode = False 

        self.sparse_linear = None
        for name, module in self.flag_model.model.named_modules():
            if 'sparse_linear' in name and isinstance(module, torch.nn.Linear):
                self.sparse_linear = module
                break
        
        if self.sparse_linear is None:
            for name, module in self.flag_model.model.named_modules():
                if isinstance(module, torch.nn.Linear) and module.out_features == 1:
                    if 'colbert' not in name: 
                        self.sparse_linear = module
                        break
        
        if self.sparse_linear is None:
            raise ValueError("Could not find 'sparse_linear' layer.")

        self.transformer = None
        from transformers import PreTrainedModel
        
        if isinstance(self.flag_model.model, PreTrainedModel):
            self.transformer = self.flag_model.model
        else:
            for name, module in self.flag_model.model.named_modules():
                if isinstance(module, PreTrainedModel) and "Linear" not in str(type(module)):
                    if module.config.hidden_size > 0:
                        self.transformer = module
                        break
        
        if self.transformer is None:
            self.transformer = self.flag_model.model # Fallback

        print(f"Moving internal modules to {self.device}...")
        self.transformer.to(self.device)
        self.sparse_linear.to(self.device)

    def set_query_mode(self, is_query: bool):
        self.is_query_mode = is_query

    def eval(self):
        self.transformer.eval()
        self.sparse_linear.eval()

    def encode(self, texts, batch_size=32, convert_to_numpy=True, show_progress_bar=True, **kwargs):
        if isinstance(texts, str): texts = [texts]
        n_samples = len(texts)

        batch_vectors = np.zeros((n_samples, self.vocab_size), dtype=np.float16)
        
        encoded_input = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=1024, 
            return_tensors='pt'
        )
        
        target_device = next(self.transformer.parameters()).device
        
        encoded_input = {k: v.to(target_device) for k, v in encoded_input.items()}
        
        input_ids = encoded_input['input_ids']

        with torch.no_grad():
            try:
                model_output = self.transformer(**encoded_input, return_dict=True)
            except TypeError:
                model_output = self.transformer(encoded_input)

            if hasattr(model_output, 'last_hidden_state'):
                last_hidden = model_output.last_hidden_state
            elif isinstance(model_output, dict) and 'token_embeddings' in model_output:
                last_hidden = model_output['token_embeddings']
            else:
                last_hidden = model_output[0]

            sparse_logits = self.sparse_linear(last_hidden) 
            sparse_weights = torch.relu(sparse_logits).squeeze(-1)

        input_ids_np = input_ids.cpu().numpy()
        weights_np = sparse_weights.float().cpu().numpy()
        
        pad_id = self.tokenizer.pad_token_id
        
        for i in range(n_samples):
            valid_mask = input_ids_np[i] != pad_id
            ids = input_ids_np[i][valid_mask]
            w = weights_np[i][valid_mask]
            np.maximum.at(batch_vectors[i], ids, w)

        return batch_vectors


class SparseIndex:
    """
    A sparse index for dense-sparse hybrid retrieval.
    It stores data as a CSR Matrix, drastically reducing memory usage (from 1.2TB to ~3GB).
    """
    def __init__(self, sparse_matrix, doc_ids):
        # sparse_matrix: scipy.sparse.csr_matrix
        self.matrix = sparse_matrix
        self.doc_ids = doc_ids # list of doc_ids
        
    def search(self, queries, k=1000):
        """
        queries: numpy array (dense) [n_queries, 250002]
        """
        # 1. Convert queries to sparse matrix
        q_sparse = sp.csr_matrix(queries)
        
        # 2. Compute Similarity (Dot Product)
        scores_matrix = q_sparse.dot(self.matrix.T) 
        
        if not isinstance(scores_matrix, np.ndarray):
             scores_matrix = scores_matrix.toarray()

        n_queries = len(queries)
        top_k_indices = np.argpartition(scores_matrix, -k, axis=1)[:, -k:]

        rows = np.arange(n_queries)[:, None]
        top_k_scores = scores_matrix[rows, top_k_indices]

        sort_indices = np.argsort(-top_k_scores, axis=1)
        
        final_indices = top_k_indices[rows, sort_indices]
        final_scores = top_k_scores[rows, sort_indices]
        
        return final_scores, final_indices

    @property
    def ntotal(self):
        return self.matrix.shape[0]

def encode_dataset_sparse(model, dataset, encode_batch_size):
    """
    Replace encode.encode_dataset_faiss with sparse matrix construction.
    """
    docs = []
    docid_to_idx = {}
    idx_to_docid = {}

    log.info("Reading all documents...")
    for i, doc in enumerate(dataset.docs_iter()):
        docid_to_idx[doc.doc_id] = i
        idx_to_docid[i] = doc.doc_id
        docs.append(doc.text if hasattr(doc, 'text') else doc.body)
    
    n_samples = len(docs)
    sparse_batches = []
    
    model.eval()
    
    iterable = tqdm(range(0, n_samples, encode_batch_size), desc="Encoding Sparse")
    
    for i in iterable:
        batch_docs = docs[i : i + encode_batch_size]
        dense_batch = model.encode(batch_docs, batch_size=encode_batch_size, convert_to_numpy=True, show_progress_bar=False)
        sparse_batch = sp.csr_matrix(dense_batch.astype(np.float32))
        sparse_batch.eliminate_zeros()
        sparse_batches.append(sparse_batch)

    log.info("Stacking sparse matrices...")
    full_matrix = sp.vstack(sparse_batches)
    
    log.info(f"Sparse Matrix Created. Shape: {full_matrix.shape}, Stored Size: {full_matrix.data.nbytes / 1024**3:.2f} GB")

    index = SparseIndex(full_matrix, list(idx_to_docid.values()))
    
    return index, (idx_to_docid, docid_to_idx)

class MultilingualE5Wrapper:
    def __init__(self, model_name, device="cuda"):
        self.device = device
        print(f"Loading Multilingual E5: {model_name}...")
        self.model = SentenceTransformer(model_name, device=device)
        self.task_instruction = "Given a web search query, retrieve relevant passages that answer the query"
        self.is_query_mode = False 

    def eval(self):
        self.model.eval()

    def set_query_mode(self, is_query: bool):
        self.is_query_mode = is_query

    def encode(self, texts, batch_size=32, convert_to_numpy=True, show_progress_bar=True, **kwargs):
        if isinstance(texts, str): texts = [texts]
        formatted_texts = []
        for t in texts:
            if self.is_query_mode:
                final_text = f'Instruct: {self.task_instruction}\nQuery: {t}'
            else:
                final_text = t
            formatted_texts.append(final_text)
        return self.model.encode(formatted_texts, batch_size=batch_size, convert_to_numpy=convert_to_numpy, show_progress_bar=show_progress_bar, normalize_embeddings=True)


class KaLMWrapper:
    def __init__(self, model_name, device="cuda"):
        self.device = device
        print(f"Loading KaLM Embedder: {model_name}...")
        
        # KaLM explicitly says DO NOT set trust_remote_code
        self.model = SentenceTransformer(
            model_name, 
            trust_remote_code=False, 
            device=device
        )
        # Model card suggests max_seq_length = 512
        self.model.max_seq_length = 512
        
        # Standard retrieval instruction
        self.task_instruction = "Given a web search query, retrieve relevant passages that answer the query"
        self.is_query_mode = False 

    def eval(self):
        self.model.eval()

    def set_query_mode(self, is_query: bool):
        self.is_query_mode = is_query

    def encode(
        self,
        texts,
        batch_size=32,
        convert_to_numpy=True,
        show_progress_bar=True,
        device=None, 
        **kwargs
    ):
        # KaLM supports the 'prompt' argument in encode()
        # Format based on usage: "Instruct: {task} \n Query: "
        prompt = None
        if self.is_query_mode:
            prompt = f"Instruct: {self.task_instruction} \n Query: "
            
        return self.model.encode(
            texts,
            prompt=prompt, # Pass the prompt here
            batch_size=batch_size,
            convert_to_numpy=convert_to_numpy,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=True
        )
    

class E5V2Wrapper:
    def __init__(self, model_name, device="cuda"):
        self.device = device
        print(f"Loading E5-v2 Embedder: {model_name}...")
        self.model = SentenceTransformer(model_name, device=device)
        self.is_query_mode = False

    def eval(self):
        self.model.eval()

    def set_query_mode(self, is_query: bool):
        """Switch mode: Query Encoding (True) -> 'query: ', Document Encoding (False) -> 'passage: '"""
        self.is_query_mode = is_query

    def encode(self, texts, batch_size=32, convert_to_numpy=True, show_progress_bar=True, **kwargs):
        if isinstance(texts, str): texts = [texts]
        formatted_texts = []
        for t in texts:
            if self.is_query_mode:
                # Queries start with "query: "
                final_text = f'query: {t}'
            else:
                # Documents start with "passage: "
                final_text = f'passage: {t}'
            formatted_texts.append(final_text)
        
        return self.model.encode(
            formatted_texts,
            batch_size=batch_size,
            convert_to_numpy=convert_to_numpy,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=True
        )


def count_parameters(model, trainable=True):
    """Returns the total number of parameters, optionally filtering only trainable parameters."""
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "train_dense", description="Trains a dense retrieval model"
    )

    parser.add_argument(
        "--data_path", default="./datasets/TREC-TOT", help="location to dataset"
    )

    parser.add_argument(
        "--negatives_path",
        default="./bm25_negatives",
        help="path to folder containing negatives ",
    )

    parser.add_argument(
        "--query", choices=["title", "text", "title_text"], default="title_text"
    )

    parser.add_argument(
        "--model_or_checkpoint",
        type=str,
        required=True,
        help="hf checkpoint/ path to pt-model",
    )
    parser.add_argument(
        "--embed_size", required=True, type=int, help="hidden size of the model"
    )
    parser.add_argument(
        "--epochs", type=int, default=0, help="number of epochs to train"
    )
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="weight decay")
    parser.add_argument("--warmup_steps", type=int, default=0, help="warmup steps")
    parser.add_argument(
        "--batch_size", type=int, default=24, help="batch size (training)"
    )
    parser.add_argument(
        "--encode_batch_size", type=int, default=124, help="batch size (inference)"
    )
    parser.add_argument(
        "--evaluation_steps",
        type=int,
        default=-1,
        help="steps before evaluation is run",
    )

    parser.add_argument(
        "--freeze_base_model",
        action="store_true",
        default=False,
        help="if set, freezes the base layer and trains only a projection layer on top",
    )
    parser.add_argument(
        "--metrics",
        required=False,
        default=run_lexicon_ja.METRICS,
        help="csv - metrics to evaluate",
    )
    parser.add_argument(
        "--n_hits", default=1000, type=int, help="number of hits to retrieve"
    )

    parser.add_argument(
        "--device", type=str, default="cuda", help="device to train /evaluate model on"
    )

    parser.add_argument(
        "--model_dir", type=str, help="folder to store model & runs", required=True
    )
    parser.add_argument(
        "--run_id", required=True, help="run id (required if run_format = trec_eval)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--negatives_out",
        default=None,
        help="if provided, dumps negatives for use in training other models",
    )
    parser.add_argument(
        "--n_negatives", default=10, type=int, help="number of negatives to obtain"
    )
    parser.add_argument(
        "--corrupt_method", default=None, help="methods to re-initialize model layers"
    )
    parser.add_argument(
        "--run_output_dir", 
        default=None, 
        help="Separate folder to save .run files (if different from model_dir)"
    )

    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    utils.set_seed(args.seed)
    log.info(f"args: {args}")

    tot_ja.register(args.data_path)
    metrics = args.metrics.split(",")

    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)

    if args.freeze_base_model:
        if args.model_or_checkpoint == "facebook/dpr-question_encoder-single-nq-base":
            model = CustomDPRModel(args.model_or_checkpoint, device=args.device)
        elif args.model_or_checkpoint == "YOUR OWN MODEL NAME":
            model = LouisDPRModel("OpenMatch/co-condenser-large-msmarco")
            model = model.to(args.device)
            model.load_model(args.model_or_checkpoint)
        elif args.model_or_checkpoint == "dense_models/baseline_distilbert_ckpt/model":
            model = SentenceTransformer(args.model_or_checkpoint, device=args.device)
            transformer_model = model._modules["0"].auto_model

            # the second last transformer layer
            second_last_layer = transformer_model.transformer.layer[4]

            if args.corrupt_method == "kaiming_normal":
                # Re-initialize weights of the attention
                init.kaiming_normal_(
                    second_last_layer.attention.q_lin.weight, nonlinearity="relu"
                )
                init.kaiming_normal_(
                    second_last_layer.attention.k_lin.weight, nonlinearity="relu"
                )
                init.kaiming_normal_(
                    second_last_layer.attention.v_lin.weight, nonlinearity="relu"
                )
                init.kaiming_normal_(
                    second_last_layer.attention.out_lin.weight, nonlinearity="relu"
                )
            elif args.corrupt_method == "xavier_normal":
                init.xavier_normal_(second_last_layer.attention.q_lin.weight)
                init.xavier_normal_(second_last_layer.attention.k_lin.weight)
                init.xavier_normal_(second_last_layer.attention.v_lin.weight)
                init.xavier_normal_(second_last_layer.attention.out_lin.weight)
            else:
                print("Passed None or invalid corrupt methods!")

            log.info(
                f"the second last transformer layer corrupted with {args.corrupt_method}"
            )

            # re-initialize biases to zero
            second_last_layer.attention.q_lin.bias.data.zero_()
            second_last_layer.attention.k_lin.bias.data.zero_()
            second_last_layer.attention.v_lin.bias.data.zero_()
            second_last_layer.attention.out_lin.bias.data.zero_()

        else:
            base_model = SentenceTransformer(
                args.model_or_checkpoint, device=args.device, trust_remote_code=True
            )
            for param in base_model.parameters():
                param.requires_grad = False
            projection = models.Dense(
                args.embed_size, args.embed_size, activation_function=nn.Tanh()
            )
            model = SentenceTransformer(
                modules=[base_model, projection], device=args.device
            )
    else:
        if "bge-m3" in args.model_or_checkpoint.lower():
            if args.embed_size > 10000: 
                 log.info(f"Detected BGE-M3 (SPARSE Mode)...")
                 model = BGEM3SparseWrapper(args.model_or_checkpoint, device=args.device)
            else: 
                 log.info(f"Detected BGE-M3 (DENSE Mode)...")
                 model = BGEM3Wrapper(args.model_or_checkpoint, device=args.device)
        elif "dpr" in args.model_or_checkpoint.lower() and "voidful" in args.model_or_checkpoint.lower():
             log.info(f"Detected Voidful DPR model...")
             model = MultilingualDPRWrapper(args.model_or_checkpoint, device=args.device)
        elif "dpr-xm" in args.model_or_checkpoint.lower():
             log.info(f"Detected DPR-XM model (Setting language to ja_XX)...")
             model = DPRXMWrapper(args.model_or_checkpoint, device=args.device)
        elif "mcontriever" in args.model_or_checkpoint.lower():
             log.info(f"Detected mContriever model...")
             model = MContrieverWrapper(args.model_or_checkpoint, device=args.device)
        elif "multilingual-e5" in args.model_or_checkpoint and "instruct" in args.model_or_checkpoint:
             log.info(f"Detected Multilingual E5 Instruct model...")
             model = MultilingualE5Wrapper(args.model_or_checkpoint, device=args.device)
        elif "KaLM" in args.model_or_checkpoint:
            log.info(f"Detected KaLM model...")
            model = KaLMWrapper(args.model_or_checkpoint, device=args.device)
        elif "granite-embedding" in args.model_or_checkpoint:
            log.info(f"Loading Granite model (Standard): {args.model_or_checkpoint}")
            model = SentenceTransformer(args.model_or_checkpoint, device=args.device)
        elif "e5-large-v2" in args.model_or_checkpoint:
             log.info(f"Detected E5-v2 model...")
             model = E5V2Wrapper(args.model_or_checkpoint, device=args.device)
        elif "Ivysaur" in args.model_or_checkpoint:
            log.info(f"Loading Ivysaur model (Tiny): {args.model_or_checkpoint}")
            model = SentenceTransformer(args.model_or_checkpoint, device=args.device)
        else:
            log.info(f"Loading standard SentenceTransformer (Default): {args.model_or_checkpoint}")
            model = SentenceTransformer(args.model_or_checkpoint, device=args.device)

    # print("Total trainable parameters:", count_parameters(model))
    # print("Total parameters (including non-trainable):", count_parameters(model, trainable=False))

    irds_splits = {}
    st_data = {}

    # splits
    for split in {"train", "dev"}:
        irds_splits[split] = ir_datasets.load(f"ja-corpus:{split}")


        # -- don't use hard negatives for now --
        # log.info(f"loaded split {split}")
        # st_data[split] = data.SBERTDataset(
        #     irds_splits[split],
        #     query_type=args.query,
        #     negatives=utils.read_json(
        #         os.path.join(
        #             args.negatives_path, f"{split}-{args.query}-negatives.json"
        #         )
        #     ),
        # )
        # -- don't use hard negatives for now --

    # log.info(f"training model for {args.epochs} epochs")
    # train_dataloader = DataLoader(st_data["train"], shuffle=True, batch_size=args.batch_size)

    # args.loss_fn = "mnrl"
    # if args.loss_fn == "mnrl":
    #     train_loss = losses.MultipleNegativesRankingLoss(model=model)
    # else:
    #     raise NotImplementedError(args.loss_fn)

    # -- skip training for now --
    # val_evaluator = data.get_ir_evaluator(
    #     st_data["dev"],
    #     name=f"dev",
    #     mrr_at_k=[1000],
    #     ndcg_at_k=[10, 1000],
    #     corpus_chunk_size=args.encode_batch_size,
    # )
    # -- skip training for now --

    # optimizer_params = {
    #     "lr": args.lr
    # }

    # Tune the model
    # model.fit(train_objectives=[(train_dataloader, train_loss)],
    #           evaluation_steps=args.evaluation_steps,
    #           output_path=os.path.join(model_dir, "model"),
    #           evaluator=val_evaluator,
    #           epochs=args.epochs,
    #           warmup_steps=args.warmup_steps,
    #           optimizer_params=optimizer_params,
    #           weight_decay=args.weight_decay,
    #           save_best_model=True)

    log.info("encoding corpus with model")
    if hasattr(model, "set_query_mode"):
        model.set_query_mode(False)

    embed_size = args.embed_size
    # index, (idx_to_docid, docid_to_idx) = encode.encode_dataset_faiss(
    #     model,
    #     embedding_size=embed_size,
    #     dataset=irds_splits["train"],
    #     device=args.device,
    #     encode_batch_size=args.encode_batch_size,
    #     model_name=args.model_or_checkpoint,
    # )
    # index, (idx_to_docid, docid_to_idx) = encode_dataset_sparse(
    #     model=model,
    #     dataset=irds_splits["train"],
    #     encode_batch_size=args.encode_batch_size
    # )
    if "bge-m3" in args.model_or_checkpoint.lower() and args.embed_size > 10000:
        log.info("Using SPARSE encoding pipeline (scipy.sparse)...")
        # 调用刚才添加的稀疏编码函数
        index, (idx_to_docid, docid_to_idx) = encode_dataset_sparse(
            model=model,
            dataset=irds_splits["train"],
            encode_batch_size=args.encode_batch_size
        )
    else:
        log.info("Using DENSE encoding pipeline (FAISS)...")
        embed_size = args.embed_size
        index, (idx_to_docid, docid_to_idx) = encode.encode_dataset_faiss(
            model,
            embedding_size=embed_size,
            dataset=irds_splits["train"],
            device=args.device,
            encode_batch_size=args.encode_batch_size,
            model_name=args.model_or_checkpoint,
        )

    runs = {}
    eval_res_agg = {}
    eval_res = {}

    try:
        log.info("attempting to load test set")
        # plug in the test set
        irds_splits["test"] = ir_datasets.load(f"ja-corpus:test")
        log.info("success!")
    except KeyError:
        log.info("couldn't find test set!")
        pass

    # [ADDED] Initialize variable to store domain-specific aggregated results
    domain_agg_results = {}

    split_qrels = {}
    for split, dataset in irds_splits.items():
        log.info(f"running & evaluating {split}")

        # [ADDED] Switch to query mode if available (Important for KaLM/E5/etc)
        if hasattr(model, "set_query_mode"):
            model.set_query_mode(True)

        run = encode.create_run_faiss(
            model=model,
            dataset=dataset,
            query_type=args.query,
            device=args.device,
            eval_batch_size=args.encode_batch_size,
            index=index,
            idx_to_docid=idx_to_docid,
            docid_to_idx=docid_to_idx,
            top_k=args.n_hits,
            model_name=args.model_or_checkpoint,
        )
        runs[split] = run

        if dataset.has_qrels():
            qrel, n_missing = utils.get_qrel(dataset, run)
            split_qrels[split] = qrel
            evaluator = pytrec_eval.RelevanceEvaluator(qrel, metrics)

            eval_res[split] = evaluator.evaluate(run)
            
            # [ADDED] Map QIDs to domains
            qid_to_domain = {}
            for q in dataset.queries_iter():
                qid_to_domain[q.query_id] = getattr(q, 'domain', 'unknown')

            # [ADDED] Calculate and log aggregated results per domain
            domain_metric_values = defaultdict(lambda: defaultdict(list))
            
            # Group scores by domain
            for qid, q_metrics in eval_res[split].items():
                domain = qid_to_domain.get(qid, "unknown")
                for metric, value in q_metrics.items():
                    domain_metric_values[domain][metric].append(value)

            # Compute mean and std for each domain
            log.info(f"=== Domain-specific Results ({split}) ===")
            domain_agg_results[split] = {}
            for domain, metric_dict in domain_metric_values.items():
                domain_agg_results[split][domain] = {}
                log.info(f"--- Domain: {domain} ---")
                for metric, values in metric_dict.items():
                    mean = np.mean(values)
                    std = np.std(values)
                    domain_agg_results[split][domain][metric] = (mean, std)
                    log.info(f"{metric:<12}: {mean:.4f} ({std:0.4f})")
            log.info("===============================")

            # Overall Aggregation
            eval_res_agg[split] = utils.aggregate_pytrec(eval_res[split], "mean")
            log.info(f"=== Overall Results ({split}) ===")
            for metric, (mean, std) in eval_res_agg[split].items():
                log.info(f"{metric:<12}: {mean:.4f} ({std:0.4f})")

    utils.write_json(
        {
            "aggregated_result": eval_res_agg,
            "domain_aggregated_result": domain_agg_results, # [ADDED] Save domain results to JSON
            "run": runs,
            "result": eval_res,
            "args": vars(args),
        },
        os.path.join(model_dir, "out.gz"),
        zipped=True,
    )

    run_id = args.run_id
    assert run_id is not None

    save_run_dir = args.run_output_dir if args.run_output_dir else model_dir
    os.makedirs(save_run_dir, exist_ok=True)

    for split, run in runs.items():
        run_path = os.path.join(save_run_dir, f"{split}.run")
        with open(run_path, "w") as writer:
            for qid, r in run.items():
                for rank, (doc_id, score) in enumerate(
                    sorted(r.items(), key=lambda _: -_[1])
                ):
                    writer.write(f"{qid}\tQ0\t{doc_id}\t{rank}\t{score}\t{run_id}\n")

    if args.negatives_out:
        log.info(f"writing negatives to folder: {args.negatives_out}")
        os.makedirs(args.negatives_out, exist_ok=True)
        out = {}

        for split, run in runs.items():
            if split == "test":
                continue
            negatives_path = os.path.join(
                args.negatives_out, f"{split}-{args.query}-negatives.json"
            )
            qrel = split_qrels[split]
            for qid, hits in run.items():
                hits = sorted(hits.items(), key=lambda _: -_[1])
                negs = []
                for (doc, score) in hits:
                    if qrel[qid].get(doc, 0) > 0:
                        continue
                    if len(negs) == args.n_negatives:
                        break
                    negs.append(doc)
                out[qid] = negs
            utils.write_json(out, negatives_path)
