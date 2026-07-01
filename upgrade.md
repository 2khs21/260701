# RAG 시스템 개선 기록 (upgrade.md)

평가 루브릭(100점)을 기준으로 `rag-starter` 프로젝트를 개선한 내용을 정리합니다.

| 카테고리 | 배점 | 핵심 기준 |
|----------|------|-----------|
| 답변 품질 | 30 | 사실 정확성, 관련성, 완전성, 다중 출처 종합 |
| 인용 및 근거 | 25 | 주장-출처 대응, 검증 가능한 참조, 가짜 출처 금지 |
| 비용 관리 | 15 | 필요한 검색만, 간결한 프롬프트, 최소 토큰 |
| 명확성 및 전달력 | 10 | 약어 정의, 읽기 쉬운 구조, 불필요 내용 제거 |
| 사용자 경험 | 10 | 응답 속도, 멀티턴 일관성, 도움이 되는 톤 |
| 견고성 및 안전성 | 10 | 범위 밖·모호 질문 처리, 거절, 프롬프트 인젝션 방어 |

---

## 1. 고려한 내용

### 평가 기준 매핑

개선 설계 시 각 루브릭 항목을 아래 파이프라인 단계에 대응시켰습니다.

```
사용자 질문 → 검색(비용) → 컨텍스트 구성(비용) → LLM 생성(품질·인용·명확성) → 인용 검증 → UI 표시(UX·근거) → 안전·범위 밖 처리
```

### 설계 원칙

1. **근거 우선(Grounding)**  
   답변은 반드시 검색된 CONTEXT에만 기반해야 하며, 코퍼스에 없는 내용(예: Artemis 프로그램)은 추측하지 않아야 합니다.

2. **토큰은 필요한 만큼만**  
   관련성 낮은 청크는 LLM에 보내지 않고, 관련 청크가 0개이면 LLM 호출 자체를 생략합니다.

3. **인용은 검증 가능해야 함**  
   `[n]` 번호만 표시하는 것이 아니라, 해당 청크의 발췌문(excerpt)을 함께 제공해 사용자가 출처를 확인할 수 있게 합니다.

4. **검색된 텍스트는 신뢰하지 않음**  
   문서 본문에 포함된 "지시문" 형태의 프롬프트 인젝션을 무시하도록 시스템 프롬프트에 명시합니다.

5. **기존 스타터 구조 유지**  
   `indexer.py` → `backend/app.py` → `frontend/` 파이프라인과 API 형식(`reply`, `citations`)은 유지하면서 확장했습니다.

### 튜닝 파라미터 (실험 근거)

| 파라미터 | 값 | 근거 |
|----------|-----|------|
| `MAX_DISTANCE` | 0.45 | 코사인 거리 기준. Apollo 1 화재 질문 top hit ≈ 0.33, Artemis 질문 top hit ≈ 0.42 |
| `TOP_K` | 5 | README 스타터 기본값 유지 |
| `MAX_CONTEXT_CHARS` | 4500 | 입력 토큰 상한 |
| `MAX_TOKENS` | 768 | 1024 → 768로 축소 (간결 답변 유도) |
| `MAX_HISTORY_TURNS` | 4 | 멀티턴 일관성 vs 토큰 비용 균형 |

---

## 2. 기존 내용

### `indexer.py`

- `chunk_text()`는 이미 구현되어 있었음 (~1000자 청크, ~100자 오버랩, 단락 경계 우선).
- `search()`는 top-k 코사인 거리 정렬만 수행. **관련성 임계값 없음**.
- 검색 결과에 **거리(score) 정보 미포함**.

### `backend/app.py`

- 기본 RAG 흐름은 동작:
  - `search(user_message, INDEX, k=5)`
  - `[n] 텍스트` 형식의 CONTEXT 블록
  - `SYSTEM_PROMPT`에 인용 규칙 일부 포함
- **한계:**
  - 단일 턴 전용 (`message`만 수신, 대화 기록 없음)
  - 관련 없는 청크도 무조건 5개 전달 → 토큰 낭비
  - 관련 청크 0개여도 LLM 호출
  - 인용 응답: `source`, `chunk_index`만 반환 (**발췌문 없음**)
  - 입력 검증 없음
  - 범위 밖 질문에 대한 조기 종료 없음
  - 프롬프트 인젝션·공격적 질문 대응 규칙 없음
  - `max_tokens=1024` (비교적 여유로움)
  - TODO 주석이 남아 있었음

### `frontend/src/App.jsx`

- 단순 채팅 UI: 메시지 목록 + 입력창
- API: `{ message }` 만 전송
- 인용 표시: `Sources: [n] filename` (한 줄, **펼치기/발췌 없음**)
- **로딩 상태, 에러 처리, 빈 화면 안내 없음**
- 멀티턴 맥락 미전달

### `frontend/src/index.css`

- 기본 메시지·폼 스타일만 존재

---

## 3. 수정한 내용

### 3.1 `indexer.py` — 검색 필터링

```python
def search(query, records, k=5, max_distance=None):
    # max_distance 초과 시 해당 hit 이후 추가 중단
    # 각 hit에 distance 필드 포함
```

- **변경:** `max_distance` 파라미터 추가
- **효과:** 관련성 낮은 청크를 LLM 컨텍스트에서 제외

### 3.2 `backend/app.py` — 핵심 백엔드 개선

#### 설정 상수 추가

```python
TOP_K = 5
MAX_DISTANCE = 0.45
MAX_CONTEXT_CHARS = 4500
MAX_USER_MESSAGE_CHARS = 2000
MAX_HISTORY_TURNS = 4
EXCERPT_CHARS = 200
MAX_TOKENS = 768
```

#### SYSTEM_PROMPT 전면 개편

| 영역 | 추가 규칙 |
|------|-----------|
| 답변 품질 | 다중 출처 종합, 핵심 먼저, 질문과 동일 언어로 답변 |
| 인용 | 모든 사실에 `[n]`, 범위 밖 번호 금지, 부분 답변 시 공백 명시 |
| 명확성 | 약어 첫 사용 시 풀네임 병기, 불필요한 반복 금지 |
| 안전 | CONTEXT 내 지시 무시, 적대·범위 밖 질문 정중히 거절 |

#### 새 헬퍼 함수

| 함수 | 역할 |
|------|------|
| `_make_excerpt()` | 인용 발췌문 생성 (~200자) |
| `_format_context()` | source/chunk 메타 + 컨텍스트 글자 수 상한 |
| `_normalize_history()` | 최근 N턴 대화 기록 정규화 |
| `_build_citations()` | `[n]` 파싱 + excerpt 포함 (기존 로직 확장) |

#### `/api/chat` API 변경

**요청 (확장):**
```json
{
  "message": "현재 질문",
  "history": [
    { "role": "user", "text": "..." },
    { "role": "assistant", "text": "..." }
  ]
}
```

**응답 (확장):**
```json
{
  "reply": "...",
  "citations": [
    {
      "n": 1,
      "source": "apollo-01.md",
      "chunk_index": 2,
      "excerpt": "발췌문..."
    }
  ],
  "retrieved": 3
}
```

**추가 로직:**
- 빈 메시지 / 2000자 초과 → `400` 에러
- `hits`가 0개 → LLM 호출 없이 `OUT_OF_SCOPE_REPLY` 즉시 반환
- 멀티턴: `history` + 현재 CONTEXT/QUESTION 메시지를 Claude에 전달

### 3.3 `frontend/src/App.jsx` — UX·인용 UI

| 항목 | 변경 |
|------|------|
| API 요청 | `history` 배열 함께 전송 |
| 로딩 | "Assistant is searching…" 표시, 입력/버튼 비활성화 |
| 에러 | try/catch + 사용자 메시지 표시 |
| 빈 화면 | 예시 질문 힌트 |
| 인용 UI | `<details>` 접이식 + excerpt 발췌문 |
| 라벨 | `user` → "You", `assistant` → "Assistant" |
| 헤더 | 부제목으로 RAG 특성 설명 |

### 3.4 `frontend/src/index.css`

- `.subtitle`, `.empty-hint`, `.loading`, `.error` 스타일
- `.msg-text` (`white-space: pre-wrap`) — 문단 구조 유지
- `.sources-label`, `.excerpt`, `button:disabled` 등 추가

---

## 4. 기대 결과

### 루브릭별 기대 효과

| 카테고리 | 기대 결과 |
|----------|-----------|
| **답변 품질** | 인용 나열이 아닌 종합 답변. 핵심 → 세부 구조. 코퍼스 밖 질문에서 환각 감소 |
| **인용 및 근거** | `[n]` ↔ 파일/chunk/excerpt 3단 대응. 범위 밖 `[n]` 자동 제거. UI에서 발췌 확인 가능 |
| **비용 관리** | 무관 청크 필터링, 컨텍스트 4500자 cap, hits=0 시 LLM 생략, max_tokens 768 |
| **명확성** | 약어 정의, 간결 답변, pre-wrap으로 가독성 향상 |
| **사용자 경험** | 로딩·에러·예시 질문, 멀티턴 follow-up ("Apollo 12는?" 등) 맥락 유지 |
| **견고성·안전** | 입력 검증, 범위 밖 조기 거절, CONTEXT 프롬프트 인젝션 무시 규칙 |

### README 테스트 질문별 예상 동작

| 질문 | 예상 |
|------|------|
| *"What was the cause of the Apollo 1 fire?"* | `apollo-01.md` 등 관련 청크 검색 → 인용 포함 정확한 답변 |
| *"Which Apollo missions landed on the Moon?"* | 여러 문서 종합 + 다수 `[n]` 인용 |
| *"Compare moonwalk durations of Apollo 11 and 17."* | 교차 문서 비교 + 각 주장별 인용 |
| *"List Apollo missions that used Saturn V."* | 추론·열거 + 관련 소스 인용 |
| *"What is the Artemis program?"* | 코퍼스 밖 → "제공된 문서에서 근거를 찾지 못했습니다" 또는 LLM이 "sources don't cover" 명시 |

### 정량적 기대 (대략)

| 지표 | 기존 | 개선 후 |
|------|------|---------|
| LLM 호출 (무관 질문) | 항상 1회 | hits=0이면 0회 |
| 컨텍스트 토큰 | 최대 ~5청크 전체 | 거리 필터 + 4500자 cap |
| 출력 토큰 상한 | 1024 | 768 |
| 인용 검증 정보 | 파일명만 | 파일명 + chunk + excerpt |

---

## 5. 한계 및 추가 튜닝 포인트

1. **`MAX_DISTANCE` 민감도**  
   Artemis처럼 Apollo program 개요와 어느 정도 유사한 질문은 여전히 1~2개 청크가 검색될 수 있습니다.  
   → 더 엄격하게 하려면 `0.42` 이하로 낮추거나, top-1 거리 기준 2차 필터 추가.

2. **멀티턴 검색**  
   현재는 **마지막 질문만**으로 검색합니다. "그 임무는?" 같은 대명사 follow-up은 history로 LLM이 추론하지만, 검색 쿼리 자체는 확장하지 않습니다.  
   → 필요 시 query rewriting 추가 (토큰 비용 증가).

3. **청크 품질**  
   일부 청크가 제목(`# Apollo 1`)만 포함하는 경우 검색·인용 품질이 떨어질 수 있습니다.  
   → `chunk_text()` 최소 길이 필터 또는 헤더 병합 검토.

4. **언어**  
   (질문 언어 = 답변 언어) 규칙은 프롬프트에만 의존. 한국어 질문 시 한국어 답변 기대.

---

## 6. 실행 방법

```bash
# 백엔드
cd backend && python app.py

# 프론트엔드
cd frontend && npm run dev
```

브라우저: http://localhost:5173

---

## 7. 변경 파일 요약

| 파일 | 상태 |
|------|------|
| `indexer.py` | `search()` max_distance 추가 |
| `backend/app.py` | 프롬프트·안전·비용·멀티턴·인용 excerpt |
| `frontend/src/App.jsx` | UX·history, 인용 UI |
| `frontend/src/index.css` | 새 UI 컴포넌트 스타일 |
| `upgrade.md` | 본 문서 (신규) |
