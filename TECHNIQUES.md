# RAG 기술·기법 정리 / Techniques Reference

CFR 기반 RAG 스타터에 적용된 기법 목록입니다. **토큰 절약**과 **검색·답변 정확도 향상**에 초점을 맞췄습니다.

This document summarizes techniques implemented in the CFR RAG starter, focused on **token savings** and **retrieval/answer accuracy**.

---

## 아키텍처 개요 / Architecture Overview

| 계층 / Layer | 주요 파일 / Files | 역할 / Role |
|---|---|---|
| 인덱싱 / Indexing | `indexer.py` | PDF 정제, § 단위 청킹, 메타데이터, 하이브리드 검색 인덱스 |
| 검색·컨텍스트 / Retrieval & Context | `backend/app.py` | 하이브리드 검색, § 주입, 컨텍스트 축소·발췌 |
| 생성 / Generation | `backend/app.py` | Claude API, 적응형 출력 한도, 스트리밍 |
| UI | `frontend/src/` | 인용·경고 표시, 토큰 사용량·절약 시각화 |

**파이프라인 / Pipeline**

```
질문 → 쿼리 확장·보강 → 하이브리드 검색 → § 주입·재랭킹
     → 노이즈·TOC 필터 → 컨텍스트 최적화 → LLM(스트리밍) → 인용·경고
```

---

## 1. 인덱싱 (정확도 기반) / Indexing (Accuracy Foundation)

잘못 인덱싱되면 검색이 빗나가고, LLM에 쓸모없는 청크가 들어가 **입력 토큰이 낭비**됩니다. 아래 기법은 **검색 품질을 올려** 이후 단계의 토큰·정확도를 동시에 개선합니다.

Poor indexing wastes input tokens on irrelevant chunks. These steps improve retrieval quality first.

### 1.1 CFR PDF 노이즈 정제 / CFR PDF Noise Cleaning

**한국어**  
- 페이지 구분선(`-- n of m --`), 머리글, 단독 페이지 번호 제거  
- 줄바꿈 하이픈·단어 분리 복구 (`comm\nand` → `command`)  
- 도트 리더(목차·표) 줄 제거 및 `_recover_paragraphs`로 본문 복구  

**English**  
- Strip page breaks, running headers, lone page numbers  
- Rejoin hyphenated line breaks across PDF lines  
- Remove dot-leader TOC/table lines and recover regulatory body text  

**효과 / Effect**  
- 정확도: §91.151 등 본문이 인덱스에서 빠지는 문제 방지  
- 토큰: 목차·표 노이즈 청크가 LLM 컨텍스트에 들어가는 것을 사전 차단  

---

### 1.2 §(Section) 경계 청킹 / Section-Aware Chunking

**한국어**  
- `§ 91.151` 형태의 조항 시작을 기준으로 분할 후 ~1000자·100자 오버랩으로 패킹  
- 긴 조항은 문단 단위로 추가 분할  

**English**  
- Split on `§` section boundaries, pack into ~1000-char chunks with ~100-char overlap  
- Further split oversized sections on paragraph boundaries  

**효과 / Effect**  
- 정확도: 조항 단위 의미 보존, BM25·벡터 매칭에 유리  
- 토큰: 한 청크에 여러 무관 조항이 섞이는 비율 감소  

---

### 1.3 청크 메타데이터 (v3) / Chunk Metadata (v3)

**한국어**  
청크별 필드: `part`, `section`, `sections[]`, `section_title`, `is_toc`, `chunk_type`  

**English**  
Per-chunk fields: `part`, `section`, `sections[]`, `section_title`, `is_toc`, `chunk_type`  

**효과 / Effect**  
- 정확도: 재랭킹·§ 매칭·TOC 필터의 근거  
- 토큰: `is_toc`로 목차 청크를 컨텍스트에서 제외 가능  

---

### 1.4 노이즈·TOC 청크 필터링 (인덱스 단계) / Noise & TOC Filtering at Index Time

**한국어**  
- `is_noise_text`: 80자 미만, 도트 리더 비율 과다 → 인덱스 제외  
- `is_toc_chunk_text`: § 참조만 많고 `(a)` 본문 없음 → TOC로 분류  

**English**  
- `is_noise_text`: too short or dot-leader heavy → drop at index  
- `is_toc_chunk_text`: many § refs without operative `(a)` paragraphs → mark as TOC  

**효과 / Effect**  
- 정확도: 검색 상위에 목차만 오는 현상 완화  
- 토큰: 저신호 청크가 임베딩·BM25 풀을 차지하지 않음  

---

### 1.5 BM25용 메타데이터 토큰 보강 / Metadata Token Enrichment for BM25

**한국어**  
BM25 토큰 = 본문 `tokenize(text)` + `part91`, `91.151`, `section_title` 키워드  

**English**  
BM25 tokens = body tokens + `part91`, `91.151`, section title terms  

**효과 / Effect**  
- 정확도: "§91.151", "part 67" 같은 구조적 키워드 질의에 강함  
- 토큰: (간접) 관련 없는 벡터 상위만 의존할 때의 fallback 검색·재시도 감소  

---

## 2. 하이브리드 검색 / Hybrid Retrieval

### 2.1 벡터 + BM25 + RRF / Vector + BM25 + Reciprocal Rank Fusion

**한국어**  
- 임베딩: `paraphrase-multilingual-MiniLM-L12-v2` (384차원, 코사인 거리)  
- BM25: Okapi BM25 (`k1=1.5`, `b=0.75`)  
- `rrf_merge_dual`: 벡터 RRF `k=60`, BM25 RRF `k=40` (키워드 매칭 가중)  
- 상위 벡터 3건·BM25 6건은 fusion 후에도 반드시 풀에 포함 (`VECTOR_GUARANTEE`, `BM25_GUARANTEE`)  

**English**  
- Embeddings: multilingual MiniLM, cosine distance  
- BM25 with standard Okapi parameters  
- Dual RRF: lower k for BM25 to boost exact keyword hits  
- Guarantee top vector/BM25 hits remain in the candidate pool  

**효과 / Effect**  
- 정확도: 의미 유사 + 조항 번호·전문 용어 동시 커버  
- 토큰: 1차 검색 성공률↑ → fallback·넓은 `top_k` 의존 감소  

---

### 2.2 거리 임계값 + Fallback 검색 / Distance Threshold + Fallback

**한국어**  
- 1차: `max_distance` (기본 0.50)  
- 미스 시: `fallback_max_distance` (기본 0.60)로 재검색  
- UI에서 조정 가능 (`SearchParamsPanel`)  

**English**  
- Primary search capped by `max_distance`  
- Retry with wider `fallback_max_distance` if no hits  
- User-tunable via UI  

**효과 / Effect**  
- 정확도: 엄격·완화 검색의 균형  
- 토큰: 불필요하게 넓은 1차 검색을 기본으로 쓰지 않음  

---

### 2.3 재랭킹 보너스 / Metadata-Aware Reranking

**한국어**  
RRF 점수에 가산:  
- 파일명 `part67` 패턴 (+0.01~0.012)  
- `section` 메타데이터와 § 힌트 일치 (+0.05)  
- `chunk_type=regulation` (+0.004)  
- TOC 청크 (−0.015)  

**English**  
Bonus on RRF score for filename part patterns, section hint matches, regulation chunks; penalty for TOC.  

**효과 / Effect**  
- 정확도: Part·§ 의도에 맞는 청크 상위 고정  
- 토큰: 잘못된 상위 청크가 컨텍스트를 채우는 것 방지  

---

## 3. 쿼리 처리 (정확도) / Query Processing (Accuracy)

### 3.1 쿼리 보강 (Enrichment) / Query Enrichment

**한국어**  
질문 패턴에 따라 CFR 용어·조항 번호를 검색 쿼리에 추가. 예:  
- `fuel reserve` + VFR → `91.151`  
- Class B vs C → `91.131 91.130`  
- restricted area → `91.133`  
- BasicMed → `61.113 68.1 68.3`  
- 공역 등급 열거 → `71.31`~`71.71`  

**English**  
Rule-based expansion adds CFR section numbers and domain terms to the search query for known question patterns.  

**효과 / Effect**  
- 정확도: 자연어 질문과 조항 텍스트 간 어휘 불일치 완화  
- 토큰: 검색 실패·재시도·과다 청크 주입 감소  

---

### 3.2 암시적 § 힌트 / Implicit Section Hints

**한국어**  
보강 문자열에 없어도 질문 본문에서 § 후보 추출:  
- medical certificate 없이 PIC → `61.113`, `68.1`, `68.3`  
- restricted area → `91.133`  
- airspace enumeration → Part 71 조항들  

**English**  
Derive section hints from question text even when not echoed in enriched query string.  

**효과 / Effect**  
- 정확도: 후속 § 주입·그룹 병합의 트리거  
- 토큰: (간접) 필요한 청크만 타깃 주입  

---

### 3.3 대화 맥락 확장 / Conversational Query Expansion

**한국어**  
짧은 후속 질문(50자 미만)에 이전 사용자 발화를 붙여 검색  

**English**  
For short follow-ups, append the previous user turn to the search query.  

**효과 / Effect**  
- 정확도: "What about night?" 같은 후속 질의 개선  
- 토큰: 검색 실패로 인한 빈 컨텍스트·재질문 방지  

---

### 3.4 대화 이력 제한 / History Truncation

**한국어**  
LLM에 최근 **4턴**(8 메시지)만 전달, 사용자 메시지 최대 2000자  

**English**  
Cap history at 4 turns; max user message 2000 chars.  

**효과 / Effect**  
- 토큰: 입력 토큰 상한  
- 정확도: 오래된 무관 맥락으로 인한 혼선 완화  

---

## 4. § 인식 검색·주입 (정확도) / Section-Aware Retrieval & Injection

### 4.1 § 청크 그룹 (`_records_for_section`) / Section Chunk Groups

**한국어**  
PDF 분할로 조항이 여러 청크에 잘릴 때:  
1. § 헤더·본문이 있는 **앵커 청크** 선택 (짧은 제목만 있는 청크 회피)  
2. 동일 파일에서 **연속 chunk_index**를 따라가며 본문 수집 (최대 4청크)  
3. 새 § 블록이 시작되면 중단 (교차 참조 `§ 91.129 and the following`는 유지)  

**English**  
When a § spans multiple PDF chunks: pick a substantive anchor, walk consecutive chunks in the same file, stop at unrelated § boundaries.  

**효과 / Effect**  
- 정확도: Class B/C 비교 등에서 조항 **헤더만** LLM에 가는 문제 해결  
- 토큰: 비교 질문에만 그룹 확장; 단순 질문은 1~2청크 유지  

---

### 4.2 § 힌트 주입 (`_inject_section_hits`) / Section Hint Injection

**한국어**  
검색이 §를 놓치면 인덱스에서 그룹을 끌어와 `distance=0.26`, `section_hint_match`로 삽입  
- Part 68(BasicMed): `part67.pdf` 내 PART 68 블록 일괄 주입  
- §91.133: 다중 조항이 묶인 청크에 `focus_sections`로 본문만 추출  

**English**  
If search misses a hinted §, inject the section group from the full index with boosted scores. Special handling for Part 68 block and §91.133 focus extraction.  

**효과 / Effect**  
- 정확도: 벤치마크 §91.131/130, BasicMed, restricted area 개선  
- 토큰: 전체 인덱스 재검색 대신 타깃 청크만 추가  

---

### 4.3 비교·열거 질문 시 그룹 병합 / Group Merge for Compare & Enumerate

**한국어**  
- 복수 § 힌트: 라운드로빈으로 청크 배치 (한 조항이 컨텍스트를 독점하지 않도록)  
- 공역 등급 열거: `part71`만 유지, `part73` 노이즈 제거  
- 비교·다중 § 질문: `max_context_chars` 최대 2배 (최대 10,000자)  

**English**  
- Round-robin merge for multi-§ compare questions  
- Part 71-only filter for airspace class enumeration  
- Expanded context budget for complex multi-section questions  

**효과 / Effect**  
- 정확도: 양쪽 조항·모든 등급이 컨텍스트에 들어갈 확률↑  
- 토큰: 단순 질문에는 확장 미적용; 열거 시 part73 제거로 낭비 절감  

---

### 4.4 § 블록 포커스 (`_hit_context_text`) / Section Block Focus

**한국어**  
한 청크에 여러 §가 있을 때 `focus_sections` 또는 메타 `section`으로 해당 § 구간만 LLM에 전달  

**English**  
Extract only the relevant § block when a chunk bundles multiple sections (e.g. §91.225 + §91.133).  

**효과 / Effect**  
- 정확도: 잘못된 § 본문이 컨텍스트에 섞이는 것 방지  
- 토큰: 불필요한 동일 청크 내 다른 조항 텍스트 제외  

---

## 5. 컨텍스트 최적화 (토큰 절약) / Context Optimization (Token Savings)

### 5.1 노이즈·TOC 필터 (`_prepare_hits`) / Noise & TOC Filtering

| 기법 / Technique | 조건 / When | 절약 / Saves |
|---|---|---|
| `filtered_noise_chunks` | 도트 리더·표 청크 제거 | 해당 청크 문자 수 |
| `filtered_toc_chunks` | 실질 본문 ≥3건일 때 TOC 제거 | TOC 청크 문자 수 |
| `reduced_top_k` | 강한 매칭 + 순위 gap 큼 → 상위 4건만 | 나머지 청크 |

**한국어**  
복잡·다중 § 질문에서는 `reduced_top_k`를 **건너뜀** (정확도 우선).  

**English**  
Skip `reduced_top_k` for complex / multi-§ questions to protect accuracy.  

---

### 5.2 § 포커스 트리밍 (`focused_top_k`) / §-Focused Trimming

**한국어**  
- 단일 § 힌트 + 강한 매칭: 같은 § 청크만 최대 2건  
- 거리 gap이 큰 1위 앵커: 동일 § 또는 1청크로 축소  

**English**  
For single-§ strong matches, keep at most 2 same-section chunks or collapse to one anchor hit.  

**효과 / Effect**  
- 토큰: 단순 사실 질문에서 `top_k=6` 전부를 LLM에 넣지 않음  
- 정확도: 복잡 질문은 트리밍 완화·그룹 확장으로 상충 방지  

---

### 5.3 키워드 발췌 (`context_excerpts`) / Keyword Excerpts

**한국어**  
단순·단일 §·강한 매칭·총 900자 이상일 때만:  
질문 키워드가 많이 들어간 문단만 추출 (최대 ~550자/청크)  

**English**  
For simple single-§ questions with strong match and long chunks, send keyword-focused paragraph excerpts instead of full text.  

**효과 / Effect**  
- 토큰: 입력 컨텍스트 30~50% 축소 가능  
- 정확도: 복잡·비교·다중 § 질문에서는 **비활성화**  

---

### 5.4 컨텍스트·출력 한도 / Context & Output Caps

| 파라미터 / Parameter | 기본값 / Default | 목적 / Purpose |
|---|---|---|
| `max_context_chars` | 4500 | LLM 입력 상한 |
| `max_tokens` | 768 | 출력 상한 |
| `_adaptive_max_tokens` | 짧은 질문 → 512 | 출력 토큰 절약 |
| 복잡 질문 | 최대 1024 출력 | 열거·비교 답 품질 |

**한국어**  
`max_context_chars`는 UI에서 500~12,000 조정 가능.  

**English**  
Context char budget is user-tunable in the search params panel.  

---

### 5.5 컨텍스트 헤더 / Compact Context Headers

**한국어**  
각 청크 앞에 `[n] | §91.151 | 제목… | filename.pdf` 형태의 짧은 헤더  

**English**  
Numbered compact headers tie citations `[n]` to sources without repeating full metadata in prose.  

**효과 / Effect**  
- 정확도: 인용 번호와 소스 매핑  
- 토큰: 헤더는 짧게 유지 (제목 72자 절단)  

---

## 6. LLM 생성 / LLM Generation

### 6.1 시스템 프롬프트 설계 / System Prompt Design

**한국어 (요지)**  
- CONTEXT만 사용, `[n]` 인용 필수  
- CONTEXT에 없는 Part·문서 언급 금지  
- 간결히, 질문 반복·미사용 컨텍스트 재서술 금지  
- 질문과 같은 언어로 답변  

**English (summary)**  
- Ground answers in CONTEXT only; mandatory `[n]` citations  
- Do not cite parts/documents absent from CONTEXT  
- Be concise; answer in the user's language  

**효과 / Effect**  
- 정확도: 환각·외부 규정 인용 억제  
- 토큰: 불필요한 서론·반복 감소  

---

### 6.2 스트리밍 응답 / Streaming Response

**한국어**  
- `POST /api/chat` 기본 `stream: true`  
- NDJSON: `retrieval` → `delta`* → `done` (인용·경고·usage 포함)  
- 검색 완료 후 LLM 토큰만 스트리밍 (검색 지연은 선행)  

**English**  
- Default streaming via NDJSON events  
- Retrieval metadata first, then text deltas, then final payload with citations/warnings  

**효과 / Effect**  
- 토큰 절약과 무관, **체감 지연(latency)** 개선  
- `done`까지 동일한 후처리(인용·경고) 유지  

---

### 6.3 검색 0건 시 LLM 스킵 / No-Hit LLM Skip

**한국어**  
검색 결과가 없으면 LLM 호출 없이 `OUT_OF_SCOPE_REPLY` 반환  

**English**  
Skip the LLM call when retrieval returns zero hits.  

**효과 / Effect**  
- 토큰: 입력·출력 API 비용 0  
- 정확도: 근거 없는 답 생성 방지  

---

## 7. 사후 품질·안전 / Post-Response Quality & Safety

### 7.1 응답 경고 (`_assess_response`) / Response Warnings

| 코드 / Code | 의미 / Meaning |
|---|---|
| `no_retrieval` | 검색 0건 |
| `noise_only_retrieval` | 노이즈 청크만 검색 |
| `fallback_retrieval` | fallback 거리로 재검색 |
| `weak_retrieval` | best distance > 0.42 |
| `insufficient_context` | 답변에 “근거 부족” 표현 감지 |
| `missing_citations` | 긴 답인데 `[n]` 없음 |
| `external_reference` | 답의 Part X가 검색 청크에 없음 |

**한국어**  
약한 검색이어도 LLM은 호출하고, UI에 경고로 사용자에게 알림 (이전 LLM 스킵 방식 대체).  

**English**  
Weak retrieval still produces an LLM answer; warnings surface reliability issues in the UI.  

**효과 / Effect**  
- 정확도: 사용자가 답 신뢰도 판단 가능  
- 토큰: 스킵 대신 경고 → 답은 시도, 비용은 발생  

---

### 7.2 인용 추출 / Citation Extraction

**한국어**  
답변의 `[n]`을 파싱해 실제 사용된 청크만 `citations` 배열로 반환 (발췌 200자)  

**English**  
Parse `[n]` markers and return only cited chunks with short excerpts.  

**효과 / Effect**  
- 정확도: 답변·출처 대조 가능  
- 토큰: UI에 전체 청크를 실을 필요 없음  

---

### 7.3 토큰 절약 추적 / Token Savings Tracking

**한국어**  
각 최적화 단계(`focused_top_k`, `context_excerpts`, `reduced_top_k` 등)마다 `tokens_saved` 추정치를 누적해 UI `TokenUsageBar`에 표시  

**English**  
Estimate and surface per-step token savings from context optimizations in the UI.  

**효과 / Effect**  
- 운영: 어떤 기법이 컨텍스트를 줄였는지 가시화  

---

## 8. 요약 매트릭스 / Summary Matrix

| 기법 / Technique | 토큰 절약 / Token Savings | 정확도 향상 / Accuracy |
|---|:---:|:---:|
| PDF 노이즈 정제·§ 청킹 | ○ (간접) | ●●● |
| 하이브리드 검색 (Vector+BM25+RRF) | ○ | ●●● |
| 쿼리 보강·§ 힌트 | ○ | ●●● |
| § 청크 그룹·주입 | △ (확장 시 증가) | ●●● |
| 노이즈·TOC 필터 | ●● | ●● |
| focused_top_k / reduced_top_k | ●● | ● (단순 질문) |
| context_excerpts | ●● | ● (단순 질문) |
| adaptive max_tokens | ● | ○ |
| 대화 이력·컨텍스트 상한 | ●● | ○ |
| § 블록 포커스 | ● | ●● |
| 시스템 프롬프트·인용 강제 | ● | ●●● |
| 응답 경고 | — | ●● (신뢰도) |
| 검색 0건 LLM 스킵 | ●●● | ●● |
| 스트리밍 | — (UX) | — |

**범례 / Legend:** ●●● 강함 / strong · ●● 중간 / medium · ● 약함 / mild · ○ 간접 / indirect · △ 트레이드오프 / trade-off  

---

## 9. 관련 파일 / Related Files

| 파일 / File | 설명 / Description |
|---|---|
| `indexer.py` | 인덱싱, 하이브리드 검색, 메타데이터 |
| `backend/app.py` | RAG 파이프라인, 컨텍스트 관리, API |
| `frontend/src/App.jsx` | 채팅 UI, 스트리밍 수신 |
| `frontend/src/SearchParamsPanel.jsx` | 검색·컨텍스트 파라미터 |
| `frontend/src/TokenUsageBar.jsx` | 토큰·절약 표시 |
| `frontend/src/WarningBlock.jsx` | 검색·답변 경고 |
| `scripts/run_benchmark.py` | 회귀 벤치마크 (`stream: false`) |

---

## 10. 재인덱싱·실행 / Reindex & Run

```bash
# 인덱스 재생성 (PDF·청킹 변경 후)
python indexer.py

# 백엔드
cd backend && python app.py

# 프론트엔드
cd frontend && npm run dev
```

**한국어**  
`indexer.py`의 청킹·정제 로직을 바꾼 뒤에는 반드시 `python indexer.py`로 `index.pkl`을 다시 만들어야 검색·§ 그룹이 반영됩니다.  

**English**  
After changing chunking or PDF normalization in `indexer.py`, rebuild `index.pkl` before expecting retrieval or §-group fixes to take effect.  
