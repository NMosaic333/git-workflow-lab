import os
import uuid
import time
from typing import List, Dict, Any, Optional
from fastapi import UploadFile

from backend.utils.pdf_parser import parse_pdf
from backend.utils.chunker import chunk_document
from backend.retrieval.hybrid_search import HybridRetriever
from backend.rag.pipeline import RAGPipeline
from backend.evaluation.tracker import EvaluationTracker

class RAGService:
    def __init__(self):
        self.upload_dir = os.getenv("UPLOAD_DIR", "./data/uploaded_papers")
        os.makedirs(self.upload_dir, exist_ok=True)
        
        self.retriever = HybridRetriever()
        self.pipeline = RAGPipeline()
        self.tracker = EvaluationTracker()

    async def process_pdf_upload(self, file: UploadFile) -> Dict[str, Any]:
        """Save PDF, parse, chunk, and add to retrievers."""
        # Save file
        file_path = os.path.join(self.upload_dir, file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
            
        paper_id = str(uuid.uuid4())
        paper_title = file.filename
        
        # Parse PDF
        pages_data = parse_pdf(file_path)
        
        # Chunk text
        chunks = chunk_document(pages_data, paper_title, paper_id)
        
        # Add to vector store and BM25
        self.retriever.add_chunks(chunks)
        
        return {
            "status": "success",
            "paper_id": paper_id,
            "title": paper_title,
            "num_pages": len(pages_data),
            "num_chunks": len(chunks)
        }

    def get_all_papers(self) -> List[Dict[str, Any]]:
        return self.retriever.get_all_papers()

    def delete_paper(self, paper_id: str):
        self.retriever.delete_paper(paper_id)
        return {"status": "success", "paper_id": paper_id}

    def handle_query(self, query: str, mode: str, filter_paper_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Process user query, retrieve chunks, generate answer, and log metrics."""
        # 1. Retrieve
        start_retrieval = time.time()
        retrieved_chunks = self.retriever.search(query, top_k=5, filter_paper_ids=filter_paper_ids)
        retrieval_latency = time.time() - start_retrieval
        
        # 2. Generate
        answer, generation_latency = self.pipeline.generate_response(query, retrieved_chunks, mode)
        
        # 3. Log Evaluation Metrics
        # Rough token estimation (1 token ~= 4 chars)
        input_tokens = sum(len(c.get("text", "")) for c in retrieved_chunks) // 4 + len(query) // 4
        output_tokens = len(answer) // 4
        
        self.tracker.log_query(
            query=query,
            mode=mode,
            retrieval_latency=retrieval_latency,
            generation_latency=generation_latency,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            num_chunks=len(retrieved_chunks)
        )
        
        return {
            "answer": answer,
            "source_chunks": retrieved_chunks,
            "metrics": {
                "retrieval_latency_ms": round(retrieval_latency * 1000, 2),
                "generation_latency_ms": round(generation_latency * 1000, 2)
            }
        }
        
    def get_metrics(self) -> Dict[str, Any]:
        return self.tracker.get_metrics()
