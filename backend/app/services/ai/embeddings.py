from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.core.config import settings


@lru_cache
def get_embeddings() -> Embeddings:
    """The one place the app builds its embedding model.

    Chunks and questions must be embedded by the same model, or their vectors can't be compared.
    Gemini embeds them differently on purpose: embed_documents uses task type RETRIEVAL_DOCUMENT,
    embed_query uses RETRIEVAL_QUERY (a short question is matched against longer passages).
    """
    return GoogleGenerativeAIEmbeddings(model=settings.embedding_model, google_api_key=settings.gemini_api_key)
