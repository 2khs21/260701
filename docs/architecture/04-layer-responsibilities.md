# 레이어별 책임 / Layer Responsibilities

4개 계층(인덱싱 → 검색·컨텍스트 → 생성 → UI)의 역할 분리입니다.

```mermaid
graph TB
    subgraph L1["인덱싱 계층 · indexer.py"]
        L1A[PDF 정제]
        L1B[§ 단위 청킹]
        L1C[메타데이터 v3]
        L1D[BM25 + Vector 인덱스]
    end

    subgraph L2["검색·컨텍스트 계층 · backend/app.py"]
        L2A[쿼리 확장]
        L2B[하이브리드 검색]
        L2C[§ 주입·재랭킹]
        L2D[컨텍스트 축소·발췌]
    end

    subgraph L3["생성 계층 · backend/app.py"]
        L3A[SYSTEM_PROMPT]
        L3B[적응형 max_tokens]
        L3C[Claude 스트리밍]
        L3D[인용 검증·경고]
    end

    subgraph L4["UI 계층 · frontend/src/"]
        L4A[채팅 + 스트리밍]
        L4B[인용·경고 표시]
        L4C[토큰 사용량·절약]
        L4D[검색 파라미터 패널]
    end

    L1 --> L2 --> L3 --> L4
```

## 계층별 주요 파일

| 계층 | 경로 | 핵심 컴포넌트 |
|------|------|---------------|
| 인덱싱 | `indexer.py` | `build_index`, `hybrid_search` |
| 검색·컨텍스트 | `backend/app.py` | `_retrieve_merged`, `_optimize_for_llm_context` |
| 생성 | `backend/app.py` | `_chat_stream`, `_finalize_chat_response` |
| UI | `frontend/src/` | `App.jsx`, `TokenUsageBar`, `GlossaryPanel` |

[← 목록으로](./README.md)
