# src/agent/schema.py

from typing import List, Optional, Dict, Any
from pydantic import BaseModel


class AgentState(BaseModel):
    # Latest user message
    query: str

    # Parameters extracted from all messages so far
    extracted_params: Optional[Dict[str, Any]] = None

    # Which parameters are still missing
    missing_fields: Optional[List[str]] = None

    # Whether we need to ask the user for more info
    needs_more_info: bool = False

    # Results from crop recommendation tool
    crop_results: Optional[List[Dict[str, Any]]] = None

    # Retrieved RAG documents
    rag_results: Optional[List[Dict[str, Any]]] = None

    # Final answer or follow-up question to user
    answer: Optional[str] = None

