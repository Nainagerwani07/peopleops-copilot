from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter

from app.core.config import settings


@lru_cache
def get_chat_model() -> BaseChatModel:
    """The one place the app builds its LLM. Switch provider/model via LLM_PROVIDER / LLM_MODEL."""
    return init_chat_model(
        settings.llm_model,
        model_provider=settings.llm_provider,
        api_key=settings.gemini_api_key,
        temperature=0,  # deterministic-ish: routing and SQL should not be creative
        # Free tier allows few requests/minute: wait our turn instead of getting HTTP 429.
        rate_limiter=InMemoryRateLimiter(requests_per_second=settings.llm_requests_per_minute / 60, max_bucket_size=1),
    )
