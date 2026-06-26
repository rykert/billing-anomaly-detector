from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True)

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/billing_anomaly"

    # Azure OpenAI (blank = use local fallback)
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_deployment_gpt4o: str = "gpt-4o"
    azure_openai_deployment_embedding: str = "text-embedding-3-small"

    # Local fallback
    use_local_models: bool = True
    ollama_base_url: str = "http://localhost:11434"

    # Detection
    anomaly_threshold: float = 0.80


@lru_cache
def get_settings() -> Settings:
    return Settings()
