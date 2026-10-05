from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from app.core.config import settings


@lru_cache
def get_chat_model() -> BaseChatModel:
    """The one place the app builds its LLM. Switch provider/model via LLM_PROVIDER / LLM_MODEL."""
    return init_chat_model(
        settings.llm_model,
        model_provider=settings.llm_provider,
        api_key=settings.gemini_api_key,
        temperature=0,  # deterministic-ish: routing and SQL should not be creative
    )
