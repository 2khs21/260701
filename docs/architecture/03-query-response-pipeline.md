# 질의·응답 파이프라인 / Query-Response Pipeline

`POST /api/chat` 요청이 검색·생성·응답까지 처리되는 온라인 흐름입니다.

```mermaid
flowchart TD
    Q[사용자 질문] --> Greet{인사말?}
    Greet -->|Yes| Skip1["검색·LLM 생략"]
    Greet -->|No| Expand["쿼리 확장·보강<br/>_expand_query · _enrich_search_query"]

    Expand --> Retrieve["하이브리드 검색<br/>_retrieve_merged"]

    subgraph Hybrid["indexer.hybrid_search"]
        V["Vector Search<br/>cosine distance"]
        B["BM25 Search<br/>메타 토큰 포함"]
        RRF["RRF 융합<br/>guarantee slots"]
    end

    Retrieve --> Hybrid
    Hybrid --> Hints["§ 힌트 추출<br/>_implicit_section_hints"]
    Hints --> Inject["§ 주입<br/>_inject_section_hits"]
    Inject --> Rerank["재랭킹·필터<br/>_rerank_hits · TOC 제거"]
    Rerank --> Prep["_prepare_hits<br/>거리·집중도 기반 트림"]
    Prep --> Opt["_optimize_for_llm_context<br/>발췌·노이즈 제거"]
    Opt --> Ctx["_format_context<br/>헤더 + 발췌문"]
    Ctx --> LLM[Claude 스트리밍]
    LLM --> Post["_finalize_chat_response<br/>인용·용어집·경고"]

    Retrieve -->|0 hits| OOS["범위 밖 응답<br/>LLM 생략"]
    Skip1 --> UI[Frontend 표시]
    Post --> UI
    OOS --> UI

    subgraph StreamEvents["NDJSON 이벤트"]
        E1["type: retrieval"]
        E2["type: delta"]
        E3["type: done"]
    end

    LLM -.-> StreamEvents
```

## 스트리밍 이벤트 순서

1. `retrieval` — 검색 쿼리, 히트 수, retrieval 요약
2. `delta` — LLM 토큰 스트리밍 (반복)
3. `done` — 최종 reply, citations, warnings, usage, timing

## 토큰 절약 분기

| 조건 | 동작 |
|------|------|
| 인사말 | 검색·LLM 모두 생략 |
| 검색 0건 | LLM 생략, 범위 밖 응답 |
| 강한 매칭 | 청크 수·컨텍스트 축소 |
| 짧은 질문 | `max_tokens` 512로 축소 |

[← 목록으로](./README.md)
