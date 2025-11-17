"""
Configuration loader for Jarvis
Loads settings from .env file using pydantic for validation
"""

import os
from pathlib import Path
from typing import Literal, Optional
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class LLMConfig(BaseModel):
    """LLM Configuration"""
    provider: Literal["deepseek", "openai", "ollama"] = Field(
        default="deepseek",
        description="LLM provider"
    )

    # DeepSeek
    deepseek_api_key: Optional[str] = Field(default=None, alias="DEEPSEEK_API_KEY")
    deepseek_model: str = Field(default="deepseek-chat", alias="DEEPSEEK_MODEL")
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com",
        alias="DEEPSEEK_BASE_URL"
    )

    # OpenAI
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")

    # Ollama
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        alias="OLLAMA_BASE_URL"
    )
    ollama_model: str = Field(default="llama3.1", alias="OLLAMA_MODEL")

    class Config:
        populate_by_name = True


class LEANNConfig(BaseModel):
    """LEANN (RAG) Configuration"""
    embedding_model: str = Field(
        default="facebook/contriever",
        alias="EMBEDDING_MODEL"
    )
    embedding_device: Literal["cuda", "cpu"] = Field(
        default="cuda",
        alias="EMBEDDING_DEVICE"
    )
    top_k: int = Field(default=5, alias="TOP_K", ge=1, le=20)
    chunk_size: int = Field(default=512, alias="CHUNK_SIZE", ge=128, le=2048)
    chunk_overlap: int = Field(default=50, alias="CHUNK_OVERLAP", ge=0, le=256)
    similarity_threshold: float = Field(
        default=0.5,
        alias="SIMILARITY_THRESHOLD",
        ge=0.0,
        le=1.0
    )

    class Config:
        populate_by_name = True


class WebSearchConfig(BaseModel):
    """Web Search Configuration"""
    enabled: bool = Field(default=True, alias="WEB_SEARCH_ENABLED")
    brave_api_key: Optional[str] = Field(default=None, alias="BRAVE_API_KEY")
    max_results: int = Field(
        default=3,
        alias="WEB_SEARCH_MAX_RESULTS",
        ge=1,
        le=10
    )

    class Config:
        populate_by_name = True


class ASRConfig(BaseModel):
    """Automatic Speech Recognition Configuration"""
    enabled: bool = Field(default=False, alias="ASR_ENABLED")
    model: str = Field(default="omniASR_CTC_1B", alias="ASR_MODEL")
    device: Literal["cuda", "cpu"] = Field(default="cuda", alias="ASR_DEVICE")
    language: str = Field(default="auto", alias="ASR_LANGUAGE")

    class Config:
        populate_by_name = True


class TTSConfig(BaseModel):
    """Text-to-Speech Configuration"""
    enabled: bool = Field(default=False, alias="TTS_ENABLED")
    model: str = Field(default="xtts_v2", alias="TTS_MODEL")
    device: Literal["cuda", "cpu"] = Field(default="cuda", alias="TTS_DEVICE")
    language: str = Field(default="en", alias="TTS_LANGUAGE")
    streaming: bool = Field(default=True, alias="TTS_STREAMING")
    speaker_wav: Optional[str] = Field(default=None, alias="TTS_SPEAKER_WAV")

    @validator("speaker_wav")
    def validate_speaker_wav(cls, v):
        if v and not Path(v).exists():
            raise ValueError(f"Speaker WAV file not found: {v}")
        return v

    class Config:
        populate_by_name = True


class StorageConfig(BaseModel):
    """Storage Configuration"""
    users_dir: Path = Field(default=Path("users"), alias="USERS_DIR")
    max_upload_size_mb: int = Field(
        default=50,
        alias="MAX_UPLOAD_SIZE_MB",
        ge=1,
        le=500
    )

    @validator("users_dir")
    def create_users_dir(cls, v):
        v.mkdir(parents=True, exist_ok=True)
        return v

    class Config:
        populate_by_name = True


class SystemConfig(BaseModel):
    """System Configuration"""
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        alias="LOG_LEVEL"
    )
    gpu_batch_size: int = Field(
        default=32,
        alias="GPU_BATCH_SIZE",
        ge=1,
        le=256
    )

    class Config:
        populate_by_name = True


class Config(BaseModel):
    """Main Jarvis Configuration"""
    llm: LLMConfig
    leann: LEANNConfig
    web_search: WebSearchConfig
    asr: ASRConfig
    tts: TTSConfig
    storage: StorageConfig
    system: SystemConfig

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables"""
        return cls(
            llm=LLMConfig(
                provider=os.getenv("LLM_PROVIDER", "deepseek"),
                **{k: v for k, v in os.environ.items() if k.startswith(("DEEPSEEK_", "OPENAI_", "OLLAMA_"))}
            ),
            leann=LEANNConfig(
                **{k: v for k, v in os.environ.items() if k.startswith("EMBEDDING_") or k in ["TOP_K", "CHUNK_SIZE", "CHUNK_OVERLAP", "SIMILARITY_THRESHOLD"]}
            ),
            web_search=WebSearchConfig(
                **{k: v for k, v in os.environ.items() if k.startswith("WEB_SEARCH_") or k == "BRAVE_API_KEY"}
            ),
            asr=ASRConfig(
                **{k: v for k, v in os.environ.items() if k.startswith("ASR_")}
            ),
            tts=TTSConfig(
                **{k: v for k, v in os.environ.items() if k.startswith("TTS_")}
            ),
            storage=StorageConfig(
                **{k: v for k, v in os.environ.items() if k in ["USERS_DIR", "MAX_UPLOAD_SIZE_MB"]}
            ),
            system=SystemConfig(
                **{k: v for k, v in os.environ.items() if k in ["LOG_LEVEL", "GPU_BATCH_SIZE"]}
            )
        )

    def validate(self) -> list[str]:
        """Validate configuration and return list of warnings/errors"""
        issues = []

        # Check LLM API keys
        if self.llm.provider == "deepseek" and not self.llm.deepseek_api_key:
            issues.append("⚠️  DEEPSEEK_API_KEY not set")
        elif self.llm.provider == "openai" and not self.llm.openai_api_key:
            issues.append("⚠️  OPENAI_API_KEY not set")

        # Check web search
        if self.web_search.enabled and not self.web_search.brave_api_key:
            issues.append("⚠️  BRAVE_API_KEY not set (web search will be disabled)")

        # Check voice features
        if self.asr.enabled:
            issues.append("ℹ️  ASR enabled - make sure omnilingual-asr is installed")
        if self.tts.enabled:
            issues.append("ℹ️  TTS enabled - make sure TTS package is installed")

        # Check GPU availability
        if self.leann.embedding_device == "cuda" or self.asr.device == "cuda" or self.tts.device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    issues.append("⚠️  CUDA not available, falling back to CPU")
            except ImportError:
                issues.append("⚠️  PyTorch not installed")

        return issues


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance"""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def reload_config():
    """Reload configuration from environment"""
    global _config
    load_dotenv(override=True)
    _config = Config.from_env()
    return _config


if __name__ == "__main__":
    # Test configuration loading
    config = get_config()
    print("Configuration loaded successfully!")
    print(f"\nLLM Provider: {config.llm.provider}")
    print(f"Embedding Model: {config.leann.embedding_model}")
    print(f"Web Search: {'Enabled' if config.web_search.enabled else 'Disabled'}")
    print(f"ASR: {'Enabled' if config.asr.enabled else 'Disabled'}")
    print(f"TTS: {'Enabled' if config.tts.enabled else 'Disabled'}")

    # Validate
    issues = config.validate()
    if issues:
        print("\n⚠️  Configuration Issues:")
        for issue in issues:
            print(f"  {issue}")
