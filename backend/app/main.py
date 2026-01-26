"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    settings = get_settings()
    print(f"🚀 Market Wizard starting...")
    print(f"   LLM Model: {settings.llm_model}")
    print(f"   Embedding: {settings.embedding_provider} ({settings.embedding_model})")
    yield
    # Shutdown
    print("👋 Market Wizard shutting down...")


app = FastAPI(
    title="Market Wizard API",
    description=(
        "Market and product analyzer using SSR (Semantic Similarity Rating) methodology. "
        "Based on arxiv:2510.08338v3."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "market-wizard"}


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Market Wizard API",
        "version": "0.1.0",
        "docs": "/docs",
        "methodology": "SSR (Semantic Similarity Rating)",
        "paper": "https://arxiv.org/abs/2510.08338",
    }


# Import and include routers
from app.routers import simulations, projects

app.include_router(simulations.router, prefix="/api/v1", tags=["simulations"])
app.include_router(projects.router, prefix="/api/v1", tags=["projects"])
