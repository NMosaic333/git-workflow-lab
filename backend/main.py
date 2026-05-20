import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.router import router

app = FastAPI(
    title="Research Paper RAG Assistant API",
    description="Backend API for uploading and querying research papers.",
    version="1.0.0"
)

# CORS middleware for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

@app.get("/")
def read_root():
    return {"message": "Research Paper RAG Assistant API is running"}

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("BACKEND_HOST", "0.0.0.0")
    port = int(os.getenv("BACKEND_PORT", 8000))
    uvicorn.run(app, host=host, port=port)
