import structlog
from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.response import error_response, success_response
from app.db.session import get_db
from app.models.employee import Employee
from app.schemas.chat import AIChatRequest, ChatMessageCreate, ChatSessionCreate
from app.services.ai.audit import log_ai_event
from app.services.ai.intent import classify_intent
from app.services.ai.permissions import Action, PermissionDenied, ensure
from app.services.ai.policy_rag import answer_question
from app.services.auth import get_current_user

router = APIRouter()
logger = structlog.get_logger()

# Route-level permission per intent. HR_ACTION has no single Action (apply leave vs approve leave),
# so the action agent checks per tool instead (M4).
INTENT_ACTION: dict[str, Action] = {
    "POLICY_QA": Action.ASK_POLICY,
    "SQL_QUERY": Action.QUERY_HR_DATA,
}

UNKNOWN_ANSWER = "I can help with HR policies, people and project data, and HR tasks such as leave and tickets."


def ai_unavailable() -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=error_response("AI_UNAVAILABLE", "The assistant is temporarily unavailable. Please try again shortly."),
    )


async def answer_policy(db: AsyncSession, user: Employee, message: str) -> dict | JSONResponse:
    """Run the policy RAG and audit the outcome. Caller has already checked ASK_POLICY."""
    try:
        result = await answer_question(message)
    except Exception:
        logger.exception("ai_policy_rag_failed", user_id=user.id)
        await log_ai_event(db, user=user, message=message, status="ERROR", intent="POLICY_QA", tool_name="policy_rag")
        return ai_unavailable()
    await log_ai_event(
        db, user=user, message=message, status="SUCCESS" if result.found else "NO_ANSWER",
        intent="POLICY_QA", tool_name="policy_rag",
    )
    return result.model_dump()


@router.post("/policy")
async def chat_policy(
    payload: AIChatRequest,
    current_user: Employee = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Policy RAG directly, without intent classification: auth → permission → RAG → audit."""
    try:
        ensure(current_user.role, Action.ASK_POLICY)
    except PermissionDenied as exc:
        await log_ai_event(db, user=current_user, message=payload.message, status="DENIED", intent="POLICY_QA")
        return JSONResponse(status_code=status.HTTP_403_FORBIDDEN, content=error_response("AI_PERMISSION_DENIED", str(exc)))
    data = await answer_policy(db, current_user, payload.message)
    return data if isinstance(data, JSONResponse) else success_response(data)


@router.post("/router")
async def chat_router(
    payload: AIChatRequest,
    current_user: Employee = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Classify the message, check permission, hand off to an agent, audit every outcome."""
    try:
        result = await classify_intent(payload.message)
    except Exception:
        # Full error goes to server logs only; the user gets a generic message (no internals leaked).
        logger.exception("ai_intent_failed", user_id=current_user.id)
        await log_ai_event(db, user=current_user, message=payload.message, status="ERROR", tool_name="intent_classifier")
        return ai_unavailable()

    action = INTENT_ACTION.get(result.intent)
    if action is not None:
        try:
            ensure(current_user.role, action)
        except PermissionDenied as exc:
            await log_ai_event(db, user=current_user, message=payload.message, status="DENIED", intent=result.intent)
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content=error_response("AI_PERMISSION_DENIED", str(exc)),
            )

    if result.intent == "POLICY_QA":
        data = await answer_policy(db, current_user, payload.message)
        if isinstance(data, JSONResponse):
            return data
        return success_response({"intent": result.intent, "confidence": result.confidence, **data})

    # Agents arrive in M3 (SQL) and M4 (HR actions).
    answer = UNKNOWN_ANSWER if result.intent == "UNKNOWN" else f"The {result.intent} agent is not built yet."
    await log_ai_event(db, user=current_user, message=payload.message, status="SUCCESS", intent=result.intent)
    return success_response({"intent": result.intent, "confidence": result.confidence, "answer": answer})


@router.post("/sessions")
async def create_chat_session(
    payload: ChatSessionCreate,
    current_user: Employee = Depends(get_current_user),
):
    _ = payload
    _ = current_user
    return JSONResponse(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        content=error_response("CHAT_NOT_IMPLEMENTED", "Chat session creation is a Phase-3 stub and not implemented yet"),
    )


@router.post("/sessions/{session_id}/messages")
async def post_chat_message(
    session_id: str,
    payload: ChatMessageCreate,
    current_user: Employee = Depends(get_current_user),
):
    _ = session_id
    _ = payload
    _ = current_user
    return JSONResponse(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        content=error_response("CHAT_NOT_IMPLEMENTED", "Chat messaging is a Phase-3 stub and not implemented yet"),
    )
