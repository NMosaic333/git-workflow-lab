import os
import json
import nltk
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any, Optional

# Download NLTK data if needed (for tokenization)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class BM25SearchManager:
    def __init__(self, persist_path: str = "./data/vector_store/bm25_corpus.json"):
        self.persist_path = persist_path
        self.corpus: List[Dict[str, Any]] = []
        self.bm25: Optional[BM25Okapi] = None
        self._load_corpus()

    def _load_corpus(self):
        if os.path.exists(self.persist_path):
            with open(self.persist_path, 'r', encoding='utf-8') as f:
                self.corpus = json.load(f)
            self._build_index()

    def _save_corpus(self):
        os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
        with open(self.persist_path, 'w', encoding='utf-8') as f:
            json.dump(self.corpus, f)

    def _tokenize(self, text: str) -> List[str]:
        return nltk.word_tokenize(text.lower())

    def _build_index(self):
        if not self.corpus:
            self.bm25 = None
            return
            
        tokenized_corpus = [self._tokenize(doc["text"]) for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """Add new chunks to the BM25 corpus."""
        if not chunks:
            return
            
        for chunk in chunks:
            self.corpus.append({
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "paper_id": chunk["paper_id"],
                "paper_title": chunk["paper_title"],
                "page_number": chunk["page_number"]
            })
            
        self._save_corpus()
        self._build_index()

    def delete_paper(self, paper_id: str):
        """Remove a paper from the BM25 corpus."""
        self.corpus = [doc for doc in self.corpus if doc["paper_id"] != paper_id]
        self._save_corpus()
        self._build_index()

    def search(self, query: str, top_k: int = 5, filter_paper_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Search using BM25."""
        if not self.bm25 or not self.corpus:
            return []
            
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Combine scores with corpus
        scored_docs = []
        for i, doc in enumerate(self.corpus):
            if filter_paper_ids and doc["paper_id"] not in filter_paper_ids:
                continue
            
            # Normalize BM25 score roughly (BM25 scores can be > 1, so this is just a relative score)
            score = scores[i]
            if score > 0:
                scored_docs.append({
                    "chunk_id": doc["chunk_id"],
                    "text": doc["text"],
                    "metadata": {
                        "paper_id": doc["paper_id"],
                        "paper_title": doc["paper_title"],
                        "page_number": doc["page_number"]
                    },
                    "score": score
                })
                
        # Sort by score descending
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        return scored_docs[:top_k]
