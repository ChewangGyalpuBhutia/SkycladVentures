from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Union, Dict, List, Optional
import pandas as pd
import os
from dotenv import load_dotenv
from integrated_query import IntegratedQueryEngine

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="Data Query API with RAG", version="1.0.0")

# Global query engine
query_engine = None


class QueryRequest(BaseModel):
    query: str
    query_type: str = "auto"  # auto, row, numeric, column, stats, info, rag
    use_rag: bool = False


class ChatRequest(BaseModel):
    message: str


class QueryResponse(BaseModel):
    success: bool
    data: Union[List[Dict], Dict, float, str]
    message: str = ""
    rows_returned: int = 0
    query_type_used: str = ""
    rag_used: bool = False


class ChatResponse(BaseModel):
    success: bool
    response: str
    message: str = ""


@app.on_event("startup")
async def startup_event():
    """Initialize the query engine on startup"""
    global query_engine
    try:
        print("🚀 Starting Data Query API with RAG...")
        print(f"📁 Loading dataset from: dataset/dataset.csv")
        
        # Check if API key is available
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key:
            print("🔑 GEMINI_API_KEY found in environment")
        else:
            print("⚠️  GEMINI_API_KEY not found - RAG features will be disabled")
        
        query_engine = IntegratedQueryEngine(file_path="dataset/dataset.csv")
        print("✅ Query engine initialized successfully")
        
        if query_engine.rag_enabled:
            print("🤖 RAG functionality enabled")
        else:
            print("📊 Running with traditional query features only")
            
    except Exception as e:
        print(f"❌ Failed to initialize query engine: {e}")


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Data Query API with RAG",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "/query": "Main query endpoint (supports both traditional and RAG queries)",
            "/chat": "Chat with your data using AI (requires RAG)",
            "/overview": "Get comprehensive dataset overview",
            "/suggestions": "Get example queries",
            "/health": "Health check endpoint"
        },
        "features": {
            "traditional_queries": True,
            "rag_enabled": query_engine.rag_enabled if query_engine else False,
            "supported_formats": ["CSV", "Excel", "JSON"]
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if query_engine is None:
        return {"status": "unhealthy", "message": "Query engine not initialized"}
    
    return {
        "status": "healthy",
        "query_engine": "initialized",
        "rag_enabled": query_engine.rag_enabled,
        "dataset_shape": query_engine.df.shape,
        "api_key_present": bool(os.getenv('GEMINI_API_KEY'))
    }


@app.post("/query", response_model=QueryResponse)
async def query_data(request: QueryRequest):
    """Main query endpoint"""
    if query_engine is None:
        raise HTTPException(status_code=500, detail="Query engine not initialized")

    try:
        result = query_engine.query(request.query, request.query_type, request.use_rag)
        print(result)
        
        # Determine if RAG was used
        rag_used = isinstance(result, dict) and "response" in result

        if isinstance(result, pd.DataFrame):
            if result.empty:
                return QueryResponse(
                    success=True,
                    data=[],
                    message="No data found matching your query",
                    rows_returned=0,
                    query_type_used=request.query_type,
                    rag_used=False
                )
            else:
                data = result.to_dict("records")
                return QueryResponse(
                    success=True,
                    data=data,
                    message=f"Found {len(result)} matching records",
                    rows_returned=len(result),
                    query_type_used=request.query_type,
                    rag_used=False
                )

        elif isinstance(result, dict):
            if "response" in result:  # RAG response
                response_data = {
                    "ai_response": result["response"],
                    "context_docs": result.get("num_context_docs", 0)
                }
                if "data" in result:
                    response_data["data"] = result["data"]
                    response_data["data_rows"] = result["data_rows"]
                
                return QueryResponse(
                    success=True,
                    data=response_data,
                    message="AI response generated successfully",
                    rows_returned=result.get("data_rows", 0),
                    query_type_used="rag",
                    rag_used=True
                )
            else:  # Traditional dict response
                return QueryResponse(
                    success=True,
                    data=result,
                    message="Statistics/information retrieved successfully",
                    query_type_used=request.query_type,
                    rag_used=False
                )

        elif isinstance(result, (int, float)):
            return QueryResponse(
                success=True,
                data=float(result),
                message="Calculation completed successfully",
                query_type_used=request.query_type,
                rag_used=False
            )

        else:
            return QueryResponse(
                success=True, 
                data=str(result), 
                message="Query executed successfully",
                query_type_used=request.query_type,
                rag_used=rag_used
            )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Query error: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat_with_data(request: ChatRequest):
    """Chat with your data using AI"""
    if query_engine is None:
        raise HTTPException(status_code=500, detail="Query engine not initialized")
    
    if not query_engine.rag_enabled:
        raise HTTPException(
            status_code=503, 
            detail="RAG functionality not available. Please check GEMINI_API_KEY in .env file."
        )

    try:
        response = query_engine.chat_with_data(request.message)
        return ChatResponse(
            success=True,
            response=response,
            message="Chat response generated successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Chat error: {str(e)}")


@app.get("/overview")
async def get_overview():
    """Get dataset overview"""
    if query_engine is None:
        raise HTTPException(status_code=500, detail="Query engine not initialized")
    
    try:
        overview = query_engine.get_data_overview()
        return {"success": True, "data": overview}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Overview error: {str(e)}")


@app.get("/suggestions")
async def get_suggestions():
    """Get query suggestions"""
    if query_engine is None:
        raise HTTPException(status_code=500, detail="Query engine not initialized")
    
    try:
        suggestions = query_engine.suggest_queries()
        return {"success": True, "suggestions": suggestions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Suggestions error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("🌟 Starting Data Query API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)