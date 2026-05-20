from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from typing import List, Optional
from pydantic import BaseModel
from backend.services.rag_service import RAGService

router = APIRouter()

# Dependency to get RAGService
def get_rag_service():
    # In a real app, this should be a singleton managed by FastAPI lifecycle
    if not hasattr(get_rag_service, "service"):
        get_rag_service.service = RAGService()
    return get_rag_service.service

class QueryRequest(BaseModel):
    query: str
    mode: str = "general"
    filter_paper_ids: Optional[List[str]] = None

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...), rag_service: RAGService = Depends(get_rag_service)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    try:
        result = await rag_service.process_pdf_upload(file)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/papers")
async def list_papers(rag_service: RAGService = Depends(get_rag_service)):
    return rag_service.get_all_papers()

@router.delete("/papers/{paper_id}")
async def delete_paper(paper_id: str, rag_service: RAGService = Depends(get_rag_service)):
    return rag_service.delete_paper(paper_id)

@router.post("/query")
async def handle_query(request: QueryRequest, rag_service: RAGService = Depends(get_rag_service)):
    try:
        result = rag_service.handle_query(
            query=request.query,
            mode=request.mode,
            filter_paper_ids=request.filter_paper_ids
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics")
async def get_metrics(rag_service: RAGService = Depends(get_rag_service)):
    return rag_service.get_metrics()
