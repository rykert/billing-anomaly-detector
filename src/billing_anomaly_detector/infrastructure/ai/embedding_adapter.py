from langchain_ollama import OllamaEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
from pydantic import SecretStr

from billing_anomaly_detector.domain.ports import EmbeddingPort
from billing_anomaly_detector.infrastructure.config import Settings


class AzureOpenAIEmbeddingAdapter(EmbeddingPort):
    def __init__(self, settings: Settings) -> None:
        self._embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=SecretStr(settings.azure_openai_api_key),
            azure_deployment=settings.azure_openai_deployment_embedding,
            api_version=settings.azure_openai_api_version,
        )

    async def embed(self, text: str) -> list[float]:
        return await self._embeddings.aembed_query(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return await self._embeddings.aembed_documents(texts)


class OllamaEmbeddingAdapter(EmbeddingPort):
    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self._embeddings = OllamaEmbeddings(
            base_url=base_url,
            model="nomic-embed-text",
        )

    async def embed(self, text: str) -> list[float]:
        return await self._embeddings.aembed_query(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return await self._embeddings.aembed_documents(texts)


def build_embedding_adapter(settings: Settings) -> EmbeddingPort:
    if settings.azure_openai_api_key and settings.azure_openai_endpoint:
        return AzureOpenAIEmbeddingAdapter(settings)
    return OllamaEmbeddingAdapter(settings.ollama_base_url)
