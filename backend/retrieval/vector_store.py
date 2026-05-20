import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from langchain_community.embeddings import HuggingFaceEmbeddings

class VectorStoreManager:
    def __init__(self, persist_directory: str = "./data/vector_store"):
        self.persist_directory = persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # We will use HuggingFace embeddings
        self.embedding_model_name = "all-MiniLM-L6-v2"
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="research_papers",
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """Add document chunks to the vector store."""
        if not chunks:
            return
            
        ids = [chunk["chunk_id"] for chunk in chunks]
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [
            {
                "paper_id": chunk["paper_id"],
                "paper_title": chunk["paper_title"],
                "page_number": chunk["page_number"]
            }
            for chunk in chunks
        ]
        
        # Generate embeddings
        embeddings_list = self.embeddings.embed_documents(texts)
        
        # Add to ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings_list,
            metadatas=metadatas,
            documents=texts
        )
        
    def delete_paper(self, paper_id: str):
        """Delete all chunks associated with a specific paper_id."""
        self.collection.delete(
            where={"paper_id": paper_id}
        )

    def search(self, query: str, top_k: int = 5, filter_paper_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Semantic search in the vector store."""
        query_embedding = self.embeddings.embed_query(query)
        
        where_clause = None
        if filter_paper_ids:
            if len(filter_paper_ids) == 1:
                where_clause = {"paper_id": filter_paper_ids[0]}
            else:
                where_clause = {"paper_id": {"$in": filter_paper_ids}}
                
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_clause,
            include=["documents", "metadatas", "distances"]
        )
        
        formatted_results = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                # Calculate similarity score from cosine distance
                # Chroma returns cosine distance (0 means identical). 
                # We can convert it to similarity (1 - distance)
                distance = results['distances'][0][i]
                similarity = 1.0 - distance
                
                formatted_results.append({
                    "chunk_id": results['ids'][0][i],
                    "text": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "score": similarity
                })
                
        return formatted_results

    def get_all_papers(self) -> List[Dict[str, Any]]:
        """Get a list of unique papers currently in the vector store."""
        # This is a bit inefficient for large DBs but fine for a portfolio project
        results = self.collection.get(include=["metadatas"])
        papers = {}
        for meta in results['metadatas']:
            paper_id = meta['paper_id']
            if paper_id not in papers:
                papers[paper_id] = meta['paper_title']
        
        return [{"paper_id": k, "title": v} for k, v in papers.items()]
