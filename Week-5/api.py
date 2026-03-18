from fastapi import FastAPI
from pydantic import BaseModel
from src.rag_engine import ConstitutionRAG
from typing import List

app = FastAPI(title="Nepal Constitution AI")
rag = ConstitutionRAG()

class Query(BaseModel):
    text: str

@app.post("/query")
def query_knowledge_graph(q: Query):
    results = rag.search(q.text, top_k=3)
    
    if not results:
        return {"found": False, "results": []}
    
    return {
        "found": True,
        "results": results # Return the list directly
    }