from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: str = "ollama"
    llm_model: str = "qwen2.5:7b"
    ollama_host: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text"

    qdrant_url: str = "http://localhost:6333"
    stac_url: str = "https://earth-search.aws.element84.com/v1"
    max_cloud_cover: float = 20.0
    max_area_km2: float = 100.0


settings = Settings()
