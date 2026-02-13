import json
from typing import NamedTuple, Dict, List, Any, Optional
from pathlib import Path
import ir_datasets
from ir_datasets.formats import TrecQrels, BaseDocs, BaseQueries
from ir_datasets.indices import PickleLz4FullStore
import logging

NAME = "en-corpus"

log = logging.getLogger(__name__)

class TrecToTDoc(NamedTuple):
    doc_id: str
    url: str
    title: str
    text: str
    domains: str
    id_en: Optional[str]
    popularity: int
    doc_length: int

    def to_dict(self):
        return self._asdict()


class TrecToTQuery(NamedTuple):
    query_id: str
    source: str
    llm: str
    domain: str
    rel_doc_id: str
    title: str      # "rel_doc_title"
    text: str       # "query"

    def to_dict(self):
        return self._asdict()


class TrecToTDocs(BaseDocs):

    def __init__(self, dlc):
        super().__init__()
        self._dlc = dlc

    def docs_iter(self):
        return iter(self.docs_store())

    def _docs_iter(self):
        with self._dlc.stream() as stream:
            for line in stream:
                data = json.loads(line)
                
                # Get the raw values first
                title_raw = data.get("title", None)
                text_raw = data.get("text", None)
                
                yield TrecToTDoc(
                    doc_id=data["id"],
                    url=data.get("url", None),
                    title=title_raw,
                    text=text_raw,
                    domains=data.get("domains", None),
                    id_en=data.get("id_en", None), 
                    popularity=data.get("popularity", None),
                    doc_length=data.get("doc_length", None),
                )

    def docs_cls(self):
        return TrecToTDoc

    def docs_store(self, field='doc_id'):
        return PickleLz4FullStore(
            path=f'{ir_datasets.util.home_path()}/{NAME}/docs.pklz4',
            init_iter_fn=self._docs_iter,
            data_cls=self.docs_cls(),
            lookup_field=field,
            index_fields=[field],
        )

    def docs_count(self):
        return self.docs_store().count()

    def docs_namespace(self):
        return f'{NAME}/{self._name}'

    def docs_lang(self):
        return 'en'


class LocalFileStream:
    def __init__(self, path):
        self._path = path

    def stream(self):
        return open(self._path, "rb")


class TrecToTQueries(BaseQueries):
    def __init__(self, name, dlc):
        super().__init__()
        self._name = name
        self._dlc = dlc

    def queries_iter(self):
        with self._dlc.stream() as stream:
            for line in stream:
                data = json.loads(line)

                title_raw = data.get("rel_doc_title", None)
                query_raw = data.get("query", None)
                
                yield TrecToTQuery(
                    query_id=data["query_id"],
                    source=data.get("source", None),
                    llm=data.get("llm", None),
                    domain=data.get("domain", None),
                    rel_doc_id=data.get("rel_doc_id", None),
                    title=title_raw,
                    text=query_raw
                )

    def queries_cls(self):
        return TrecToTQuery

    def queries_namespace(self):
        return f'{NAME}/{self._name}'

    def queries_lang(self):
        return 'en'


def register(path):
    # This is your original register function.
    qrel_defs = {
        1: 'answer',
        0: 'not answer',
    }
    path = Path(path)
    corpus = path / "corpus.jsonl"
    if not corpus.exists():
        log.error(f"Corpus file not found at: {corpus}")
        return

    for split in {"train", "dev", "test"}:
        name = split
        queries = path / split / "queries.jsonl"
        if not queries.exists():
            log.warning(f"not loading '{split}' split: {queries} not found")
            continue
        components = [
            TrecToTDocs(LocalFileStream(corpus)),
            TrecToTQueries(name, LocalFileStream(queries)),
        ]
        if split != "test" or (path / split / "qrel.txt").exists():
            qrel = path / split / "qrel.txt"
            if qrel.exists():
                components.append(TrecQrels(LocalFileStream(qrel), qrel_defs))
            else:
                log.warning(f"qrel.txt not found for split '{split}'")
        ds = ir_datasets.Dataset(
            *components
        )
        ir_datasets.registry.register(NAME + ":" + name, ds)
        log.info(f"registered: {NAME}:{name}")


if __name__ == '__main__':
    # Datasets path
    path = 'YOUR_DATASET_PATH_HERE'  # Replace with your dataset path

    register(path.strip())
    sets = []
    for split in {"train", "dev", "test"}:
        name = split
        sets.append(NAME + ":" + name) 
    print(f"available sets: {sets}")
    q_example = None
    for name in sets:
        try:
            dataset = ir_datasets.load(name)
        except KeyError:
            print(f"error loading {name}, skipping!")
            continue
        n_q = 0
        for q in dataset.queries_iter():
            n_q += 1
            if q_example is None:
                q_example = q
        if "test" not in name or dataset.has_qrels():
            n_qrel = 0
            for qrel in dataset.qrels_iter():
                n_qrel += 1
            print(f"{name}: n_queries={n_q}, n_qrels={n_qrel}")
        else:
             print(f"{name}: n_queries={n_q} (no qrels)")
        print()
    
    if q_example:
        print(f"example query: {q_example.to_dict()}")
    else:
        print("no queries loaded")

    n_docs = 0
    doc_example = None
    try:
        dataset = ir_datasets.load(f"{NAME}:train") 
        for doc in dataset.docs_iter():
            n_docs += 1
            if doc_example is None:
                doc_example = doc
        if doc_example:
            print(f"example doc: {doc_example.to_dict()}")
        print("corpus size (from train set iter): ", n_docs)
    except KeyError:
        print(f"Could not load {NAME}:train to count docs")
    except Exception as e:
        print(f"Error iterating docs: {e}")
