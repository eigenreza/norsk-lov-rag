"""Settings for every stage, read from a YAML file with defaults that work out of the box."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_CONFIG = Path("configs/default.yaml")


class PathsConfig(BaseModel):
    data_dir: Path = Path("data")

    @property
    def raw(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed(self) -> Path:
        return self.data_dir / "processed"

    @property
    def llm(self) -> Path:
        return self.data_dir / "llm"


class LovdataConfig(BaseModel):
    """Lovdata's open datasets, read through the public API without a key."""

    base_url: str = "https://api.lovdata.no"
    datasets: list[str] = Field(
        default_factory=lambda: [
            "gjeldende-lover.tar.bz2",
            "gjeldende-sentrale-forskrifter.tar.bz2",
        ]
    )
    user_agent: str = "norsk-lov-rag/0.1 (research)"
    timeout_seconds: float = 120.0
    max_retries: int = 4


class CorpusConfig(BaseModel):
    """Which documents make up the two corpora.

    Folders are the top-level folders of the archives: ``nl`` holds acts, ``sf`` central
    regulations, ``stv`` parliamentary decisions, ``del`` delegations and ``ins`` instructions.
    """

    full_folders: list[str] = Field(default_factory=lambda: ["nl", "sf"])
    tax_documents: list[str] = Field(default_factory=list)


class ModelLimits(BaseModel):
    """Limits of a free plan for one model. The client never goes past them."""

    requests_per_minute: int
    requests_per_day: int
    tokens_per_minute: int | None = None
    tokens_per_day: int | None = None
    refill: bool = False
    day_resets_tz: str | None = None
    counts_failed: bool = False


class ProviderConfig(BaseModel):
    style: Literal["openai", "gemini", "ollama"]
    base_url: str
    # A local server such as Ollama needs no key.
    api_key_env: str | None = None
    # Overrides the shared timeout, for slow local models.
    timeout_seconds: float | None = None


class ModelConfig(BaseModel):
    provider: str
    model: str
    limits: ModelLimits
    # Settings sent with every call to this model, such as how much it may reason.
    params: dict[str, Any] = Field(default_factory=dict)
    # The first wait after a server error. Longer when each failed call costs quota.
    retry_base_seconds: float = 2.0
    # Overrides the shared number of retries, for services that often fail for a while.
    max_retries: int | None = None


class LLMConfig(BaseModel):
    """Language models on free plans, by the role they play."""

    env_file: Path = Path(".env")
    timeout_seconds: float = 120.0
    max_retries: int = 4
    # The longest wait on a minute limit before the run stops instead.
    max_wait_seconds: float = 120.0
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    models: dict[str, ModelConfig] = Field(default_factory=dict)


class EncoderConfig(BaseModel):
    """A dense embedding model from Hugging Face, fixed to one revision."""

    hf_id: str
    revision: str
    licence: str
    query_prefix: str = ""
    passage_prefix: str = ""
    max_seq_length: int = 512
    batch_size: int = 16


class RerankerConfig(BaseModel):
    hf_id: str
    revision: str
    licence: str
    max_length: int = 512
    depth: int = 30
    batch_size: int = 8


class RetrievalConfig(BaseModel):
    """Retrieval systems compared in the evaluation, all run on the CPU."""

    model_dir: Path = Path("models/hf")
    scopes: list[str] = Field(default_factory=lambda: ["tax", "full"])
    strategies: list[str] = Field(default_factory=lambda: ["structure", "naive"])
    encoders: dict[str, EncoderConfig] = Field(default_factory=dict)
    reranker: RerankerConfig | None = None
    # Encoders run on the full corpus. The others run on the tax corpus only.
    full_corpus_encoders: list[str] = Field(default_factory=list)
    # Systems combined in the hybrid for each corpus, chosen on the development set.
    hybrid: dict[str, list[str]] = Field(default_factory=dict)
    # Planned paired comparisons: [setting/system A, setting/system B] and an optional
    # question type that limits the comparison to that type.
    comparisons: list[list[str]] = Field(default_factory=list)
    rrf_k: int = 60
    depth: int = 100
    expansion_top: int = 5


class GenerationConfig(BaseModel):
    """Answers from the retrieved text, citation checks and judging."""

    # Retrieval systems that feed the generators, as "scope-strategy/system".
    retrieval_systems: list[str] = Field(default_factory=list)
    # Model roles in llm.models that write answers.
    generators: list[str] = Field(default_factory=list)
    judge: str = "judge"
    top_k: int = 5
    max_words_per_source: int = 300
    max_context_words: int = 1200
    max_tokens: int = 512
    judge_batch_size: int = 30
    judge_max_tokens: int = 16000
    # Answers judged a second time to measure how stable the judge is.
    consistency_sample: int = 60
    seed: int = 0
    # Planned paired comparisons of correctness: [run A, run B].
    comparisons: list[list[str]] = Field(default_factory=list)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NORSK_LOV_RAG_", env_nested_delimiter="__")

    paths: PathsConfig = PathsConfig()
    lovdata: LovdataConfig = LovdataConfig()
    corpus: CorpusConfig = CorpusConfig()
    llm: LLMConfig = LLMConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    generation: GenerationConfig = GenerationConfig()


def load_settings(path: Path | None = None) -> Settings:
    """Read settings from a YAML file. A missing default file gives the built-in defaults."""
    path = path or DEFAULT_CONFIG
    if not path.exists():
        if path != DEFAULT_CONFIG:
            raise FileNotFoundError(path)
        return Settings()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Settings(**data)
