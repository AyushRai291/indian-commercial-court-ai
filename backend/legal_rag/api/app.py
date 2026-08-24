"""FastAPI application entrypoint for legal paragraph search."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Protocol

from fastapi import Depends, FastAPI, HTTPException, Request, status
from starlette.concurrency import run_in_threadpool

from legal_rag.api.schemas import HealthResponse, SearchRequest, SearchResponse
from legal_rag.api.service import SearchService, build_search_service


logger = logging.getLogger(__name__)
SERVICE_UNAVAILABLE_DETAIL = "Search service is temporarily unavailable."


class SearchExecutor(Protocol):
    """Injectable boundary used by API tests and the production service."""

    def search(self, request: SearchRequest) -> SearchResponse: ...


def create_app(
    *,
    search_service: SearchExecutor | None = None,
    service_factory: Callable[[], SearchService] = build_search_service,
) -> FastAPI:
    """Create the application with an optional prebuilt search service."""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        owned_service: SearchService | None = None
        application.state.search_service = search_service
        application.state.search_service_ready = search_service is not None
        if search_service is None:
            try:
                owned_service = await run_in_threadpool(service_factory)
                application.state.search_service = owned_service
                application.state.search_service_ready = True
            except Exception:
                logger.exception("Search service initialization failed")
        try:
            yield
        finally:
            if owned_service is not None:
                await run_in_threadpool(owned_service.close)

    application = FastAPI(
        title="Indian Commercial Court Retrieval API",
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

    return application


app = create_app()
