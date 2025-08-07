"""
FastAPI application for the Academic Search RAG database.

This module provides a REST API for querying the RAG database and checking its
health status.
"""

import os
from typing import Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from .build_rag_db import RAGDatabase

import logging

logging.basicConfig(level=logging.DEBUG)


# Initialize global context
global_context = {}


def process_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    Goes through all spans and checks whether there are any whose bounds are
    within 2 units but different. If so, it changes the bound so they match,
    making contiguous spans continuous.

    Args:
        spans: List of spans to process.

    Returns:
        List of processed spans.
    """
    processed_spans = []
    for span in spans:
        for other_span in spans:
            diff = span[1] - other_span[0]
            if abs(diff) < 2:
                span = (span[0], other_span[0])
        processed_spans.append(span)
    return processed_spans


# Pydantic Models
class QueryRequest(BaseModel):
    """Request model for the query endpoint."""

    query_text: str = Field(
        ..., description="The text to search for in the RAG database"
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of results to return",
    )
    query_kwargs: dict[str, str | int | float | bool | list[int]] | None = (
        Field(
            default=None,
            description="Additional query parameters to pass to the RAG database",
        )
    )


class Count(BaseModel):
    """Response model for the count endpoint."""

    count: list[int]
    source: list[str]


class AbstractResponse(BaseModel):
    """Response model for a paper abstract."""

    paper_id: int
    abstract: str
    title: str
    authors: list[str]
    category: str
    spans: list[tuple[int, int]]
    doi: str
    scores: list[float]
    explain_scores: list[dict[str, float]]


class QueryResponse(BaseModel):
    """Response model for the query endpoint."""

    results: list[AbstractResponse]
    total_results: int


class HealthCheck(BaseModel):
    """Response model for the health check endpoint."""

    rag_database: bool
    weaviate: bool
    status: str
    details: dict[str, Any] | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the RAG database on startup."""
    try:
        global_context["rag_db"] = RAGDatabase()
        # Test the connection
        global_context["rag_db"].connect_to_weaviate()
        global_context["rag_db"].initialize_embedding_model()
    except Exception as e:
        raise RuntimeError(f"Failed to initialize RAG database: {str(e)}")
    yield
    global_context.clear()


app = FastAPI(
    title="Academic Search API",
    description="API for querying the Academic Search RAG database",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# API Endpoints
@app.post(
    "/query",
    response_model=QueryResponse,
    summary="Query the RAG database",
    description="Search the RAG database for papers similar to the query text.",
)
async def query(request: QueryRequest):
    """
    Query the RAG database for papers similar to the given text.

    Args:
        request: The query request containing the text to search for and optional parameters.

    Returns:
        A response containing the most relevant paper chunks or abstracts.
    """
    if global_context["rag_db"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG database is not initialized",
        )

    try:
        # Convert query_kwargs values to the correct type if needed
        if request.query_kwargs is not None:
            query_args = request.query_kwargs
            processed_kwargs = {}
            for key, value in query_args.items():
                if isinstance(value, str) and value.isdigit():
                    processed_kwargs[key] = int(value)
                else:
                    processed_kwargs[key] = value
        else:
            processed_kwargs = None

        # Query the RAG database
        results = global_context["rag_db"].query_text(
            text=request.query_text,
            limit=request.limit,
            query_kwargs=processed_kwargs,
            group_by_paper=True,
        )

        # Convert results to the response model
        for paper_id in results:
            results[paper_id]["retrieved_chunks"] = sorted(
                results[paper_id]["retrieved_chunks"],
                key=lambda x: x["span"][0],
            )

        response = QueryResponse(
            results=[
                AbstractResponse(
                    paper_id=paper_id,
                    abstract=abstract["abstract"],
                    title=abstract["title"],
                    authors=abstract["authors"],
                    category=abstract["category"],
                    spans=process_spans(
                        [
                            chunk["span"]
                            for chunk in abstract["retrieved_chunks"]
                        ]
                    ),
                    scores=[
                        chunk["score"] for chunk in abstract["retrieved_chunks"]
                    ],
                    explain_scores=[
                        chunk["explain_score"]
                        for chunk in abstract["retrieved_chunks"]
                    ],
                    doi=abstract["doi"],
                )
                for paper_id, abstract in results.items()
            ],
            total_results=len(results),
        )

        return response

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error querying the RAG database: {str(e)}",
        )


@app.get(
    "/count",
    response_model=Count,
    summary="Return the number of items per source",
    description="Queries Weaviate to return the number of items per source.",
)
async def count():
    count = global_context["rag_db"].count()
    counts = {"count": list(count.values()), "source": list(count.keys())}
    return Count(**counts)


@app.get(
    "/health",
    response_model=HealthCheck,
    summary="Check the health of the API and its dependencies",
    description="Check if the API and its dependencies (RAG database, Weaviate) are running.",
)
async def health_check():
    """
    Check the health of the API and its dependencies.

    Returns:
        A health check response with the status of the API and its dependencies.
    """
    rag_status = global_context["rag_db"] is not None
    weaviate_status = False

    if rag_status:
        try:
            # Try to connect to Weaviate to check if it's running
            global_context["rag_db"].connect_to_weaviate()
            weaviate_status = True
        except Exception:
            weaviate_status = False

    overall_status = (
        "healthy" if rag_status and weaviate_status else "unhealthy"
    )

    return HealthCheck(
        rag_database=rag_status,
        weaviate=weaviate_status,
        status=overall_status,
        details={
            "weaviate_url": os.environ.get("WEAVIATE_URL", "localhost"),
            "weaviate_port": os.environ.get("WEAVIATE_PORT", 8080),
        },
    )


def main():
    import uvicorn

    PORT = int(os.environ.get("API_PORT", 8000))

    uvicorn.run(
        "academ_ai.api:app",
        host="0.0.0.0",
        port=PORT,
        reload=True,
    )


if __name__ == "__main__":
    main()
