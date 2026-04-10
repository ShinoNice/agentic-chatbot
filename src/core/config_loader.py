import yaml
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseModel):
    primary_llm: str
    embedding_model: str
    temperature: float
    max_tokens: int


class RAGSettings(BaseModel):
    raw_data_dir: str
    chunk_size: int
    chunk_overlap: int
    vector_db_path: str
    cache_dir: str
    top_k: int
    hybrid_weights: List[float]
    pinecone_index_name: str


class DoclingSettings(BaseModel):
    do_ocr: bool = True
    force_full_page_ocr: bool = False
    images_scale: float = 1.0
    min_chunks_fallback: int = 5


class AppSettings(BaseModel):
    debug_mode: bool = True
    max_iterations: int = 3
    cors_origins: List[str] = ["*"]


class RerankSettings(BaseModel):
    """Cross-encoder reranker configuration.

    `enabled` defaults to False so the eval baseline run reproduces
    today's behavior. Flip to True (or override candidate_k/top_k) for
    the post-rerank eval runs.
    """

    enabled: bool = False
    model_name: str = "BAAI/bge-reranker-base"
    candidate_k: int = 30
    top_k: int = 5

    @field_validator("top_k")
    @classmethod
    def top_k_within_candidates(cls, v, info):
        candidate_k = info.data.get("candidate_k")
        if candidate_k is not None and v > candidate_k:
            raise ValueError(
                f"rerank.top_k ({v}) must be <= rerank.candidate_k ({candidate_k})"
            )
        return v


class Settings(BaseSettings):
    """Merges .env secrets with YAML application config."""

    openai_api_key: str = Field(alias="OPENAI_API_KEY")
    pinecone_api_key: str = Field(default="", alias="PINECONE_API_KEY")
    api_key: str = Field(alias="API_KEY")

    llm: LLMSettings
    rag: RAGSettings
    docling: DoclingSettings = Field(default_factory=DoclingSettings)
    app: AppSettings = Field(default_factory=AppSettings)
    rerank: RerankSettings = Field(default_factory=RerankSettings)
    prompts: Dict[str, Any] = Field(default_factory=dict)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


def load_all_configs() -> Settings:
    """Load YAML files and .env secrets, returning a validated Settings instance."""
    root_dir = Path(__file__).parent.parent.parent
    config_dir = root_dir / "config"

    with open(config_dir / "settings.yaml", "r") as f:
        yaml_data = yaml.safe_load(f)

    prompts = {}
    for prompt_file in (config_dir / "prompts").glob("*.yaml"):
        with open(prompt_file, "r") as f:
            prompts[prompt_file.stem] = yaml.safe_load(f)

    docling_data = yaml_data.get("docling_settings", {})
    app_data = yaml_data.get("app", {})
    rerank_data = yaml_data.get("rerank_settings", {})

    return Settings(
        llm=LLMSettings(**yaml_data["model_settings"]),
        rag=RAGSettings(**yaml_data["rag_settings"]),
        docling=DoclingSettings(**docling_data),
        app=AppSettings(**app_data),
        rerank=RerankSettings(**rerank_data),
        prompts=prompts,
    )


settings = load_all_configs()
