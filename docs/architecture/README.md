# 아키텍처 다이어그램 / Architecture Diagrams

CFR RAG 스타터의 시스템 구조를 Mermaid로 정리한 문서입니다. GitHub에서 바로 렌더링됩니다.

| # | 다이어그램 | 파일 |
|---|----------|------|
| 1 | 시스템 전체 구조 | [01-system-overview.md](./01-system-overview.md) |
| 2 | 인덱싱 파이프라인 (오프라인) | [02-indexing-pipeline.md](./02-indexing-pipeline.md) |
| 3 | 질의·응답 파이프라인 (온라인) | [03-query-response-pipeline.md](./03-query-response-pipeline.md) |
| 4 | 레이어별 책임 | [04-layer-responsibilities.md](./04-layer-responsibilities.md) |
| 5 | 데이터 모델 | [05-data-model.md](./05-data-model.md) |

## 파이프라인 요약

```
질문 → 쿼리 확장·보강 → 하이브리드 검색 → § 주입·재랭킹
     → 노이즈·TOC 필터 → 컨텍스트 최적화 → LLM(스트리밍) → 인용·경고
```

## 주요 파일

| 계층 | 파일 | 역할 |
|------|------|------|
| 인덱싱 | `indexer.py` | PDF 정제, § 청킹, 메타데이터, 하이브리드 인덱스 |
| 검색·컨텍스트 | `backend/app.py` | 하이브리드 검색, § 주입, 컨텍스트 축소 |
| 생성 | `backend/app.py` | Claude API, 적응형 출력 한도, 스트리밍 |
| UI | `frontend/src/` | 인용·경고, 토큰 사용량·절약 시각화 |
