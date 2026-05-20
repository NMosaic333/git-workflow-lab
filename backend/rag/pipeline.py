import os
import time
from typing import List, Dict, Any, Tuple
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnableSequence
from backend.prompts.modes import get_prompt_for_mode

class RAGPipeline:
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("Warning: GOOGLE_API_KEY not set. Generation will fail.")
            
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro-latest", # Or gemini-pro
            google_api_key=api_key,
            temperature=0.2 # Low temperature for factual RAG
        )

    def format_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Formats the retrieved chunks into a single context string."""
        context_parts = []
        for chunk in retrieved_chunks:
            meta = chunk.get("metadata", {})
            paper_title = meta.get("paper_title", "Unknown Paper")
            page_num = meta.get("page_number", "?")
            text = chunk.get("text", "")
            
            # Format: [PaperTitle.pdf, Page X]: Text
            formatted_chunk = f"[{paper_title}, Page {page_num}]:\n{text}\n"
            context_parts.append(formatted_chunk)
            
        return "\n".join(context_parts)

    def generate_response(self, query: str, retrieved_chunks: List[Dict[str, Any]], mode: str = "general") -> Tuple[str, float]:
        """
        Generates a response using the LLM based on the retrieved context.
        Returns the answer and the generation latency.
        """
        prompt_template = get_prompt_for_mode(mode)
        context_str = self.format_context(retrieved_chunks)
        
        # Build runnable chain
        chain = prompt_template | self.llm
        
        start_time = time.time()
        try:
            response = chain.invoke({
                "context": context_str,
                "question": query
            })
            answer = response.content
        except Exception as e:
            answer = f"Error generating response: {str(e)}"
        end_time = time.time()
        
        generation_latency = end_time - start_time
        return answer, generation_latency
