from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Union, Dict, List
import pandas as pd
from integrated_query import IntegratedQueryEngine

app = FastAPI(title="Data Query API", version="1.0.0")

# Global query engine
query_engine = None


class QueryRequest(BaseModel):
    query: str
    query_type: str = "auto"  # auto, row, numeric, column, stats, info


class QueryResponse(BaseModel):
    success: bool
    data: Union[List[Dict], Dict, float, str]
    message: str = ""
    rows_returned: int = 0


@app.on_event("startup")
async def startup_event():
    """Initialize the query engine on startup"""
    global query_engine
    try:
        query_engine = IntegratedQueryEngine(file_path="dataset/dataset.csv")
        print("Query engine initialized successfully")
    except Exception as e:
        print(f"Failed to initialize query engine: {e}")


@app.post("/query", response_model=QueryResponse)
async def query_data(request: QueryRequest):
    """Main query endpoint"""
    if query_engine is None:
        raise HTTPException(status_code=500, detail="Query engine not initialized")

    try:
        result = query_engine.query(request.query, request.query_type)

        if isinstance(result, pd.DataFrame):
            if result.empty:
                return QueryResponse(
                    success=True,
                    data=[],
                    message="No data found matching your query",
                    rows_returned=0,
                )
            else:
                data = result.to_dict("records")
                return QueryResponse(
                    success=True,
                    data=data,
                    message=f"Found {len(result)} matching records",
                    rows_returned=len(result),
                )

        elif isinstance(result, dict):
            return QueryResponse(
                success=True,
                data=result,
                message="Statistics/information retrieved successfully",
            )

        elif isinstance(result, (int, float)):
            return QueryResponse(
                success=True,
                data=float(result),
                message="Calculation completed successfully",
            )

        else:
            return QueryResponse(
                success=True, data=str(result), message="Query executed successfully"
            )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Query error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
