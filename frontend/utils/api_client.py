import os
import requests
from typing import List, Dict, Any, Optional

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000/api")

def upload_pdf(file_path: str) -> Dict[str, Any]:
    url = f"{BACKEND_URL}/upload"
    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f, "application/pdf")}
        response = requests.post(url, files=files)
    response.raise_for_status()
    return response.json()

def get_papers() -> List[Dict[str, Any]]:
    url = f"{BACKEND_URL}/papers"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def delete_paper(paper_id: str) -> Dict[str, Any]:
    url = f"{BACKEND_URL}/papers/{paper_id}"
    response = requests.delete(url)
    response.raise_for_status()
    return response.json()

def query_assistant(query: str, mode: str, filter_paper_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    url = f"{BACKEND_URL}/query"
    payload = {
        "query": query,
        "mode": mode,
        "filter_paper_ids": filter_paper_ids
    }
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()

def get_metrics() -> Dict[str, Any]:
    url = f"{BACKEND_URL}/metrics"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()
