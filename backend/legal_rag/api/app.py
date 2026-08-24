"""FastAPI entrypoint for paragraph search and grounded answers."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Protocol

from fastapi import Depends, FastAPI, HTTPException, Request, status
from starlette.concurrency import run_in_threadpool

from legal_rag.api.schemas import (
    AnswerRequest,
    AnswerResponse,
    HealthResponse,
    SearchRequest,
    SearchResponse,
)
from legal_rag.api.service import SearchService, build_search_service
from legal_rag.generation import AnswerService, build_answer_service


logger = logging.getLogger(__name__)
SERVICE_UNAVAILABLE_DETAIL = "Search service is temporarily unavailable."
GENERATION_UNAVAILABLE_DETAIL = "Answer generation is temporarily unavailable."


class SearchExecutor(Protocol):
    """Injectable boundary used by API tests and the production service."""

    def search(self, request: SearchRequest) -> SearchResponse: ...


class AnswerExecutor(Protocol):
    """Injectable grounded-generation boundary used by API tests."""

    def answer(self, request: AnswerRequest) -> AnswerResponse: ...


def create_app(
    *,
    search_service: SearchExecutor | None = None,
    answer_service: AnswerExecutor | None = None,
    service_factory: Callable[[], SearchService] = build_search_service,
    answer_service_factory: Callable[[SearchExecutor], AnswerService] = (
        build_answer_service
    ),
) -> FastAPI:
    """Create the application with optional prebuilt service boundaries."""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        owned_service: SearchService | None = None
        owned_answer_service: AnswerService | None = None
        application.state.search_service = search_service
        application.state.answer_service = answer_service
        application.state.search_service_ready = search_service is not None
        if search_service is None:
            try:
                owned_service = await run_in_threadpool(service_factory)
                application.state.search_service = owned_service
                application.state.search_service_ready = True
            except Exception:
                logger.exception("Search service initialization failed")
        resolved_search_service = application.state.search_service
        if answer_service is None and resolved_search_service is not None:
            try:
                owned_answer_service = await run_in_threadpool(
                    answer_service_factory,
                    resolved_search_service,
                )
                application.state.answer_service = owned_answer_service
            except Exception:
                logger.exception("Answer service initialization failed")
        try:
            yield
        finally:
            if owned_answer_service is not None:
                await run_in_threadpool(owned_answer_service.close)
            if owned_service is not None:
                await run_in_threadpool(owned_service.close)

    application = FastAPI(
        title="Indian Commercial Court Research API",
        version="0.1.0",
        lifespan=lifespan,
    )

    def get_search_service(request: Request) -> SearchExecutor:
        service = getattr(request.app.state, "search_service", None)
        if service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=SERVICE_UNAVAILABLE_DETAIL,
            )
        return service

    def get_answer_service(request: Request) -> AnswerExecutor:
        service = getattr(request.app.state, "answer_service", None)
        if service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=GENERATION_UNAVAILABLE_DETAIL,
            )
        return service

    @application.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @application.post("/search", response_model=SearchResponse)
    async def search(
        search_request: SearchRequest,
        service: SearchExecutor = Depends(get_search_service),
    ) -> SearchResponse:
        try:
            return await run_in_threadpool(service.search, search_request)
        except Exception:
            logger.exception(
                "Search execution failed for retrieval_mode=%s",
                search_request.retrieval_mode.value,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=SERVICE_UNAVAILABLE_DETAIL,
            ) from None

    @application.post("/answer", response_model=AnswerResponse)
    async def answer(
        answer_request: AnswerRequest,
        service: AnswerExecutor = Depends(get_answer_service),
    ) -> AnswerResponse:
        try:
            return await run_in_threadpool(service.answer, answer_request)
        except Exception:
            logger.exception("Grounded answer generation failed")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=GENERATION_UNAVAILABLE_DETAIL,
            ) from None

    return application


app = create_app()
