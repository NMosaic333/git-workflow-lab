from typing import List, Dict, Any, Optional
from backend.retrieval.vector_store import VectorStoreManager
from backend.retrieval.bm25_search import BM25SearchManager

class HybridRetriever:
    def __init__(self):
        self.vector_store = VectorStoreManager()
        self.bm25_store = BM25SearchManager()

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """Add chunks to both retrievers."""
        self.vector_store.add_chunks(chunks)
        self.bm25_store.add_chunks(chunks)

    def delete_paper(self, paper_id: str):
        """Delete paper from both retrievers."""
        self.vector_store.delete_paper(paper_id)
        self.bm25_store.delete_paper(paper_id)

    def get_all_papers(self) -> List[Dict[str, Any]]:
        return self.vector_store.get_all_papers()

    def search(self, query: str, top_k: int = 5, filter_paper_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Perform hybrid search using Reciprocal Rank Fusion (RRF).
        """
        # Get results from both retrievers
        semantic_results = self.vector_store.search(query, top_k=top_k*2, filter_paper_ids=filter_paper_ids)
        keyword_results = self.bm25_store.search(query, top_k=top_k*2, filter_paper_ids=filter_paper_ids)
        
        # Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        chunk_data = {}
        
        k_rrf = 60 # Constant for RRF
        
        # Process Semantic Results
        for rank, res in enumerate(semantic_results):
            chunk_id = res["chunk_id"]
            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = res
            
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (k_rrf + rank + 1))
            
        # Process Keyword Results
        for rank, res in enumerate(keyword_results):
            chunk_id = res["chunk_id"]
            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = res
                
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (k_rrf + rank + 1))
            
        # Sort by RRF score
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        final_results = []
        for chunk_id, score in sorted_chunks[:top_k]:
            data = chunk_data[chunk_id]
            data["hybrid_score"] = score # Add the hybrid score
            final_results.append(data)
            
        return final_results
