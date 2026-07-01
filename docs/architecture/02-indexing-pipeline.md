# 인덱싱 파이프라인 / Indexing Pipeline

`python indexer.py` 실행 시 CFR 문서가 `index.pkl`로 변환되는 오프라인 흐름입니다.

```mermaid
flowchart LR
    subgraph Input["입력"]
        PDF["CFR PDF / md / txt"]
    end

    subgraph Extract["추출·정제"]
        Read[read_document]
        Clean["clean_cfr_noise<br/>페이지·머리글·하이픈 복구"]
        Norm[normalize_extracted_text]
    end

    subgraph Chunk["청킹"]
        Sec["§ 경계 분할"]
        Pack["~1000자 + 100자 오버랩"]
        Filter["is_noise_text<br/>is_toc_chunk_text"]
    end

    subgraph Meta["메타데이터 v3"]
        M["part, section, sections<br/>section_title, is_toc, chunk_type"]
        Tok["BM25 토큰 보강<br/>part91, 91.151, title"]
    end

    subgraph Embed["임베딩"]
        Model["MiniLM-L12-v2<br/>384-dim"]
        Vec["청크별 벡터"]
    end

    subgraph Output["저장"]
        PKL[("index.pkl")]
    end

    PDF --> Read --> Clean --> Norm --> Sec --> Pack --> Filter
    Filter --> M --> Tok
    Filter --> Model --> Vec
    M --> PKL
    Vec --> PKL
```

## 관련 함수 (`indexer.py`)

| 단계 | 함수 |
|------|------|
| 문서 읽기 | `read_document`, `read_pdf` |
| 노이즈 정제 | `clean_cfr_noise`, `normalize_extracted_text` |
| 청킹 | `chunk_text`, `_split_into_sections` |
| 메타데이터 | `extract_chunk_metadata`, `build_record_tokens` |
| 임베딩 | `embed`, `get_model` |
| 저장 | `build_index`, `save_index` |

[← 목록으로](./README.md)
