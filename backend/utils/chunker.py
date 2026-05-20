from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_document(pages_data: List[Dict[str, Any]], paper_title: str, paper_id: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Splits the parsed pages into smaller chunks suitable for embedding,
    while preserving metadata like page number and paper source.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = []
    chunk_id_counter = 0
    
    for page in pages_data:
        page_text = page["text"]
        page_num = page["page_number"]
        
        # Split the page text
        page_chunks = text_splitter.split_text(page_text)
        
        for text_chunk in page_chunks:
            chunks.append({
                "chunk_id": f"{paper_id}_p{page_num}_c{chunk_id_counter}",
                "paper_id": paper_id,
                "paper_title": paper_title,
                "page_number": page_num,
                "text": text_chunk
            })
            chunk_id_counter += 1
            
    return chunks
