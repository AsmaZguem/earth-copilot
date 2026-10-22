"""One place to pick the chat model, so the provider is a config change, not a code change."""

from langchain_core.language_models import BaseChatModel

from earth_copilot.config import settings


def get_chat_model(temperature: float = 0.0) -> BaseChatModel:
    provider = settings.llm_provider.lower()

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.llm_model, base_url=settings.ollama_host, temperature=temperature
        )
    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=settings.llm_model, temperature=temperature)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=settings.llm_model, temperature=temperature)

    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider!r}")
