from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI
from pydantic import SecretStr

from billing_anomaly_detector.application.services.invoice_text import (
    invoice_to_text,
)
from billing_anomaly_detector.domain.entities import Invoice
from billing_anomaly_detector.domain.ports import ExplanationPort
from billing_anomaly_detector.domain.value_objects import AnomalyScore
from billing_anomaly_detector.infrastructure.config import Settings

_SYSTEM_PROMPT = (
    "You are a healthcare billing analyst specializing in detecting unusual "
    "billing patterns. Analyze the provided claim and explain in clear, "
    "professional language why it may be anomalous compared to typical claims. "
    "Be specific about which values are unusual and why. "
    "Keep the explanation to 2-3 sentences."
)

_HUMAN_PROMPT = (
    "Anomalous claim (score: {score:.3f}, threshold: {threshold:.3f}):\n"
    "{invoice_text}\n\n"
    "Three most similar normal claims for comparison:\n"
    "{neighbor_texts}\n\n"
    "Explain why this claim appears anomalous."
)


class LangChainExplanationChain(ExplanationPort):
    def __init__(self, settings: Settings) -> None:
        llm = AzureChatOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=SecretStr(settings.azure_openai_api_key),
            azure_deployment=settings.azure_openai_deployment_chat,
            api_version=settings.azure_openai_api_version,
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_PROMPT),
            ("human", _HUMAN_PROMPT),
        ])
        self._chain = prompt | llm | StrOutputParser()
        self._threshold = settings.anomaly_threshold

    async def explain(
        self,
        invoice: Invoice,
        score: AnomalyScore,
        neighbors: list[Invoice],
    ) -> str:
        neighbor_texts = "\n".join(
            f"  - {invoice_to_text(n)}" for n in neighbors
        )
        return await self._chain.ainvoke({
            "score": score.value,
            "threshold": self._threshold,
            "invoice_text": invoice_to_text(invoice),
            "neighbor_texts": neighbor_texts,
        })


class OllamaExplanationChain(ExplanationPort):
    def __init__(self, settings: Settings) -> None:
        llm = ChatOllama(
            base_url=settings.ollama_base_url,
            model="llama3.2",
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_PROMPT),
            ("human", _HUMAN_PROMPT),
        ])
        self._chain = prompt | llm | StrOutputParser()
        self._threshold = settings.anomaly_threshold

    async def explain(
        self,
        invoice: Invoice,
        score: AnomalyScore,
        neighbors: list[Invoice],
    ) -> str:
        neighbor_texts = "\n".join(
            f"  - {invoice_to_text(n)}" for n in neighbors
        )
        return await self._chain.ainvoke({
            "score": score.value,
            "threshold": self._threshold,
            "invoice_text": invoice_to_text(invoice),
            "neighbor_texts": neighbor_texts,
        })


def build_explanation_chain(settings: Settings) -> ExplanationPort:
    if settings.azure_openai_api_key and settings.azure_openai_endpoint:
        return LangChainExplanationChain(settings)
    return OllamaExplanationChain(settings)
