import fitz  # PyMuPDF
import re
from typing import List, Dict, Any

def clean_text(text: str) -> str:
    """Clean extracted text from PDF."""
    # Replace multiple newlines with a single space or newline
    text = re.sub(r'\n+', '\n', text)
    # Remove weird unicode characters
    text = text.encode('ascii', 'ignore').decode('ascii')
    # Replace multiple spaces
    text = re.sub(r' +', ' ', text)
    return text.strip()

def parse_pdf(file_path: str) -> List[Dict[str, Any]]:
    """
    Parses a PDF file and returns a list of dictionaries, 
    where each dictionary represents a page with its text and metadata.
    """
    try:
        doc = fitz.open(file_path)
    except Exception as e:
        raise ValueError(f"Failed to open PDF {file_path}: {e}")

    pages_data = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        # Extract text blocks
        blocks = page.get_text("blocks")
        
        # Sort blocks top-to-bottom, left-to-right to handle basic multi-column
        blocks.sort(key=lambda b: (b[1], b[0]))
        
        page_text_blocks = []
        for b in blocks:
            text = b[4]
            cleaned_text = clean_text(text)
            if cleaned_text:
                page_text_blocks.append(cleaned_text)
                
        full_page_text = "\n".join(page_text_blocks)
        
        pages_data.append({
            "page_number": page_num + 1,  # 1-indexed
            "text": full_page_text
        })
        
    doc.close()
    return pages_data
