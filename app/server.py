from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agent import CustomerServiceAgent
from .config import Settings
from .llm import LLM, OpenAICompatLLM
from .mock_llm import RuleBasedLLM
from .rag import build_retriever
from .tools import build_default_tools

STATIC_DIR = Path(__file__).resolve().parent / "static"


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str = Field(default="default", min_length=1, max_length=64)


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    citations: list[dict]
    steps: list[dict]
    handoff: bool


def create_app(settings: Settings | None = None, llm: LLM | None = None) -> FastAPI:
    settings = settings or Settings.load()
    retriever = build_retriever(settings)
    tools = build_default_tools(settings, retriever)

    mode = "mock"
    if llm is None:
        if settings.use_mock_llm:
            llm = RuleBasedLLM()
        else:
            llm = OpenAICompatLLM(
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url,
                model=settings.llm_model,
                temperature=settings.llm_temperature,
            )
            mode = settings.llm_model

    agent = CustomerServiceAgent(llm, tools, max_steps=settings.max_steps)

    app = FastAPI(
        title="小岚商城智能客服 Agent",
        description="RAG + 工具调用 + 评测的客服 Agent 示例",
        version="1.0.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.agent = agent
    app.state.retriever = retriever
    app.state.settings = settings

    @app.get("/api/health")
    def health() -> dict:
        return {
            "status": "ok",
            "mode": "mock" if isinstance(llm, RuleBasedLLM) else mode,
            "chunks": len(retriever.chunks),
            "embedding": retriever._embedder.name,
        }

    @app.post("/api/chat", response_model=ChatResponse)
    def chat(payload: ChatRequest) -> ChatResponse:
        try:
            result = agent.chat(payload.session_id, payload.message)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return ChatResponse(**result.to_dict())

    @app.post("/api/sessions/{session_id}/reset")
    def reset(session_id: str) -> dict:
        agent.reset(session_id)
        return {"status": "ok", "session_id": session_id}

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
