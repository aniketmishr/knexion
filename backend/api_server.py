"""
Knexion FastAPI Backend Server

This module serves as the REST API backend for the Knexion application.
It provides endpoints for:
- File upload and processing (PDF documents)
- Conversation thread management
- Query processing through the agentic RAG workflow
- Knowledge graph visualization serving
- Chat history retrieval

The server coordinates between the Streamlit frontend and the underlying
agent workflow, handling file processing, knowledge base creation, and
response generation.
"""

import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage

# Local imports
from agent_workflow import agent, retrieve_all_threads, run_agent_query, get_conversation_history
from knowledge_store import extract_knowledge_graph, save_knowledge_graph, KnowledgeGraph, ingest_documents

# =============================================================================
# APPLICATION SETUP
# =============================================================================

app = FastAPI(
    title="Knexion API",
    description="Backend API for the Knexion Agentic Knowledge Orchestrator",
    version="1.0.0"
)

# Configuration constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
KG_CACHE_DIR = Path(".cache/graph")

# Ensure cache directory exists
KG_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# CONVERSATION MANAGEMENT ENDPOINTS
# =============================================================================

@app.get("/get-threads", response_model=List[str])
def get_all_threads() -> List[str]:
    """
    Retrieve all existing conversation thread IDs.
    
    Returns:
        List of thread IDs for conversations that have been created
    """
    try:
        return retrieve_all_threads()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving threads: {str(e)}")


@app.get("/get-msg-history", response_model=List[Dict[str, Any]])
def get_message_history(thread_id: str) -> List[Dict[str, Any]]:
    """
    Retrieve conversation history for a specific thread.
    
    Args:
        thread_id: Unique identifier for the conversation thread
    
    Returns:
        List of message dictionaries containing role, content, and metadata
    """
    try:
        return get_conversation_history(thread_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving message history: {str(e)}")

# =============================================================================
# QUERY PROCESSING ENDPOINT
# =============================================================================

@app.get("/query")
def process_query(q: str, thread_id: str) -> Dict[str, Any]:
    """
    Process a user query through the agentic RAG workflow.
    
    This endpoint:
    1. Runs the query through the complete agent workflow
    2. Returns the generated answer with associated metadata
    3. Includes knowledge graph visualization path and document context
    
    Args:
        q: User's question
        thread_id: Conversation thread identifier
    
    Returns:
        Dictionary containing answer, knowledge graph path, and document context
    """
    try:
        result = run_agent_query(q, thread_id)
        return {
            "answer": result["answer"],
            "kg_path": result["kg_path"],
            "docs": result["docs"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# =============================================================================
# FILE UPLOAD AND PROCESSING ENDPOINT
# =============================================================================

@app.post("/upload")
async def upload_documents(
    thread_id: str = Form(...),
    files: List[UploadFile] = File(...)
) -> Dict[str, str]:
    """
    Upload and process PDF documents to create a knowledge base.
    
    This endpoint performs the following steps:
    1. Validates uploaded files are PDFs
    2. Extracts text content from PDFs
    3. Splits content into manageable chunks
    4. Creates vector embeddings and stores in database
    5. Extracts knowledge graph from content
    6. Stores knowledge graph in database
    
    Args:
        thread_id: Unique identifier for the conversation thread
        files: List of uploaded PDF files
    
    Returns:
        Dictionary indicating success or failure of the operation
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Validate file types
    for file in files:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF")
    
    try:
        # Process documents for vector storage
        processed_documents = await _process_pdf_files(files)
        
        # Store documents in vector database
        print("---Storing documents in Vector Store---")
        ingest_documents(processed_documents, meta={"thread_id": thread_id})
        
        # Extract and store knowledge graph
        print("---Extracting Knowledge Graph---")
        knowledge_graph = _extract_knowledge_graph_from_documents(processed_documents)
        
        print("---Saving Knowledge Graph to TiDB---")
        save_knowledge_graph(knowledge_graph, metadata={'thread_id': thread_id})
        
        print("---Knowledge base creation completed successfully---")
        return {'result': "successful"}
        
    except Exception as e:
        print(f"Error processing upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")

# =============================================================================
# KNOWLEDGE GRAPH VISUALIZATION ENDPOINT
# =============================================================================

@app.get("/get-kg-html", response_class=HTMLResponse)
async def get_knowledge_graph_html(filename: str) -> str:
    """
    Serve knowledge graph visualization HTML files.
    
    Args:
        filename: Name of the HTML file to serve
    
    Returns:
        HTML content of the knowledge graph visualization
    
    Raises:
        HTTPException: If file is not found or invalid
    """
    file_path = KG_CACHE_DIR / filename
    
    # Security: Ensure file is within cache directory and exists
    try:
        resolved_path = file_path.resolve()
        cache_dir_resolved = KG_CACHE_DIR.resolve()
        
        # Check if file is within allowed directory
        if not str(resolved_path).startswith(str(cache_dir_resolved)):
            raise HTTPException(status_code=400, detail="Invalid file path")
        
        # Check if file exists and is a file
        if not resolved_path.exists() or not resolved_path.is_file():
            raise HTTPException(status_code=404, detail="Knowledge graph file not found")
        
        # Read and return HTML content
        return resolved_path.read_text(encoding="utf-8")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving knowledge graph: {str(e)}")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

async def _process_pdf_files(files: List[UploadFile]) -> List:
    """
    Process uploaded PDF files and extract text content.
    
    Args:
        files: List of uploaded PDF files
    
    Returns:
        List of processed document chunks
    """
    documents_storage = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP
    )
    
    print("---Creating document chunks---")
    
    for file in files:
        # Create temporary file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(await file.read())
            tmp_path = tmp_file.name
        
        try:
            # Load PDF content using LangChain
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            
            # Split documents into chunks
            doc_chunks = text_splitter.split_documents(docs)
            documents_storage.extend(doc_chunks)
            
        finally:
            # Clean up temporary file
            os.remove(tmp_path)
    
    print(f"---Processed {len(files)} files into {len(documents_storage)} chunks---")
    return documents_storage


def _extract_knowledge_graph_from_documents(documents: List) -> KnowledgeGraph:
    """
    Extract knowledge graph from processed documents.
    
    Args:
        documents: List of document chunks
    
    Returns:
        Combined knowledge graph from all documents
    """
    # Initialize empty knowledge graph
    combined_kg = KnowledgeGraph(entities=[], relationships=[])
    
    print("---Extracting entities and relationships---")
    
    # Process each document chunk
    for i, doc in enumerate(documents):
        try:
            # Extract knowledge graph from document content
            doc_kg = extract_knowledge_graph(doc.page_content)
            
            # Combine with overall knowledge graph
            combined_kg.entities.extend(doc_kg.entities)
            combined_kg.relationships.extend(doc_kg.relationships)
            
            if (i + 1) % 10 == 0:  # Progress logging
                print(f"---Processed {i + 1}/{len(documents)} documents---")
                
        except Exception as e:
            print(f"Warning: Error processing document {i + 1}: {str(e)}")
            continue
    
    print(f"---Extracted {len(combined_kg.entities)} entities and {len(combined_kg.relationships)} relationships---")
    return combined_kg

# =============================================================================
# HEALTH CHECK ENDPOINT
# =============================================================================

@app.get("/health")
def health_check() -> Dict[str, str]:
    """
    Health check endpoint for service monitoring.
    
    Returns:
        Dictionary indicating service status
    """
    return {"status": "healthy", "service": "Knexion API"}

# =============================================================================
# APPLICATION STARTUP
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("Starting Knexion API server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )