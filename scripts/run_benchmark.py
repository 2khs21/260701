#!/usr/bin/env python3
"""Run RAG benchmark questions against local /api/chat."""
import json
import re
import sys
import urllib.request

API = "http://127.0.0.1:5000/api/chat"

TESTS = [
    {
        "id": "orig-1",
        "group": "original",
        "question": (
            "What aeronautical experience is required for a private pilot certificate "
            "with an airplane single-engine rating?"
        ),
        "expect_sections": ["61.109"],
        "expect_sources": ["part61"],
        "forbid_sources": [],
        "content_checks": ["61.109", "40 total", "solo"],
    },
    {
        "id": "orig-2",
        "group": "original",
        "question": (
            "Which medical conditions disqualify an applicant for a "
            "first-class airman medical certificate?"
        ),
        "expect_sections": ["67."],
        "expect_sources": ["part67"],
        "forbid_sources": [],
        "content_checks": ["disqualif"],
    },
    {
        "id": "orig-3",
        "group": "original",
        "question": (
            "What are the fuel-reserve requirements for VFR flight, day versus night?"
        ),
        "expect_sections": ["91.151"],
        "expect_sources": ["part91"],
        "forbid_sources": [],
        "content_checks": ["91.151", "30 minute", "45 minute"],
    },
    {
        "id": "orig-4",
        "group": "original",
        "question": (
            "How do operating requirements differ between Class B and Class C airspace?"
        ),
        "expect_sections": ["91.131", "91.130"],
        "expect_sources": ["part91"],
        "forbid_sources": ["part71"],
        "content_checks": ["clearance", "two-way radio", "Class B", "Class C"],
    },
    {
        "id": "orig-5",
        "group": "original",
        "question": (
            "What must a pilot do before operating in an active restricted area?"
        ),
        "expect_sections": ["91.133"],
        "expect_sources": ["part91"],
        "forbid_sources": [],
        "content_checks": ["permission", "using or controlling", "restricted area"],
    },
    {
        "id": "bench-1",
        "group": "benchmark",
        "question": (
            "What is the distant vision standard for a first-class airman medical certificate?"
        ),
        "expect_sections": ["67.103"],
        "expect_sources": ["part67"],
        "forbid_sources": ["part61", "part91", "part71", "part73", "vol1"],
        "content_checks": ["20/20", "67.103"],
        "notes": "Single fact — only part67 in sources",
    },
    {
        "id": "bench-2",
        "group": "benchmark",
        "question": "List all classes of airspace that are designated.",
        "expect_sections": ["71."],
        "expect_sources": ["part71"],
        "forbid_sources": [],
        "content_checks": ["Class A", "Class B", "Class C", "Class D", "Class E"],
        "notes": "Enumeration — multiple chunks from part71",
    },
    {
        "id": "bench-3",
        "group": "benchmark",
        "question": (
            "Compare the distant vision standards for first-class and third-class "
            "medical certificates."
        ),
        "expect_sections": ["67.103", "67.303"],
        "expect_sources": ["part67"],
        "forbid_sources": [],
        "content_checks": ["20/20", "20/40", "67.103", "67.303"],
        "notes": "Same-file compare — need both § chunks in hits",
    },
    {
        "id": "bench-4",
        "group": "benchmark",
        "question": (
            "When can a pilot act as pilot in command without an airman medical certificate, "
            "and what are the conditions?"
        ),
        "expect_sections": ["61.113", "68."],
        "expect_sources": ["part61", "part67"],
        "forbid_sources": [],
        "content_checks": ["Part 68", "61.113", "68.", "driver"],
        "notes": "Cross-file — part61 + part67 (BasicMed in part67)",
    },
    {
        "id": "bench-5",
        "group": "benchmark",
        "question": "What are the aircraft registration requirements?",
        "expect_sections": [],
        "expect_sources": [],
        "forbid_sources": [],
        "content_checks": [],
        "expect_out_of_scope": False,
        "expect_sources": ["vol1"],
        "content_checks": ["47.1", "registration"],
        "notes": "Part 47 is in vol1.pdf — answer expected, not out-of-scope",
    },
]


def post_chat(question: str) -> dict:
    body = json.dumps({"message": question, "stream": False}).encode()
    req = urllib.request.Request(
        API,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.load(resp)


def get_hits(data: dict) -> list[dict]:
    return data.get("retrieval") or data.get("hits") or []


def source_tags(hits: list[dict]) -> list[str]:
    return sorted({h["source"] for h in hits})


def hit_blob(h: dict) -> str:
    parts = [
        h.get("text") or "",
        h.get("excerpt") or "",
        h.get("section_title") or "",
        str(h.get("section") or ""),
    ]
    return " ".join(parts)


def hit_sections(hits: list[dict]) -> list[str]:
    secs: list[str] = []
    for h in hits:
        if h.get("section"):
            secs.append(h["section"])
        for s in h.get("sections") or []:
            secs.append(s)
        for m in re.finditer(r"§\s*(\d+(?:\.\d+)*)", hit_blob(h)):
            secs.append(m.group(1))
    return list(dict.fromkeys(secs))


def evaluate(test: dict, data: dict) -> dict:
    hits = get_hits(data)
    reply = data.get("reply") or ""
    sources = source_tags(hits)
    sections = hit_sections(hits)
    lower_reply = reply.lower()
    lower_sources = " ".join(sources).lower()

    issues: list[str] = []
    passes: list[str] = []

    if test.get("expect_out_of_scope"):
        refusal_markers = (
            "don't contain", "do not contain", "not in the", "no relevant",
            "provided context", "cannot answer", "can't answer", "not found",
            "does not include", "doesn't include", "lack", "없",
        )
        refused = any(m in lower_reply for m in refusal_markers)
        if refused:
            passes.append("refused/out-of-scope reply")
        else:
            issues.append("should refuse — Part 47 not in corpus")
        if hits:
            issues.append(f"unexpected hits ({len(hits)})")
        else:
            passes.append("no retrieval hits")
        return {"passes": passes, "issues": issues, "sources": sources, "sections": sections}

    for src in test.get("expect_sources", []):
        if any(src in s.lower() for s in sources):
            passes.append(f"source contains {src}")
        else:
            issues.append(f"missing expected source pattern: {src}")

    for src in test.get("forbid_sources", []):
        if any(src in s.lower() for s in sources):
            issues.append(f"forbidden source present: {src}")

    for sec in test.get("expect_sections", []):
        if any(sec in s for s in sections) or sec in reply:
            passes.append(f"section/content {sec}")
        else:
            issues.append(f"missing expected section pattern: {sec}")

    for check in test.get("content_checks", []):
        in_reply = check.lower() in lower_reply
        in_hits = any(check.lower() in hit_blob(h).lower() for h in hits)
        if in_reply or in_hits:
            passes.append(f"content '{check}'")
        else:
            issues.append(f"missing content check: {check}")

    warnings = data.get("warnings") or []
    if warnings:
        issues.append(f"warnings: {[w.get('code') for w in warnings]}")

    return {"passes": passes, "issues": issues, "sources": sources, "sections": sections}


def main() -> int:
    results = []
    for test in TESTS:
        print(f"\n{'='*72}\n[{test['id']}] {test['question']}\n", flush=True)
        try:
            data = post_chat(test["question"])
        except Exception as exc:
            print(f"  ERROR: {exc}", flush=True)
            results.append({**test, "error": str(exc)})
            continue

        ev = evaluate(test, data)
        status = "PASS" if not ev["issues"] else "FAIL"
        print(f"  STATUS: {status}", flush=True)
        print(f"  Sources ({len(ev['sources'])}):", flush=True)
        for s in ev["sources"]:
            print(f"    - {s}", flush=True)
        print(f"  Hits: {len(get_hits(data))}", flush=True)
        for i, h in enumerate(get_hits(data), 1):
            sec = h.get("section") or "—"
            print(
                f"    [{i}] {h['source']} chunk {h['chunk_index']} "
                f"§{sec} d={h.get('distance', '?')}",
                flush=True,
            )
        if test.get("notes"):
            print(f"  Note: {test['notes']}", flush=True)
        if ev["passes"]:
            print(f"  OK: {', '.join(ev['passes'][:6])}", flush=True)
        if ev["issues"]:
            print(f"  ISSUES: {', '.join(ev['issues'])}", flush=True)
        timing = data.get("timing") or {}
        print(
            f"  Time: {timing.get('total_ms')}ms "
            f"(retrieval {timing.get('retrieval_ms')}ms, llm {timing.get('llm_ms')}ms)",
            flush=True,
        )
        print(f"  Reply preview: {(data.get('reply') or '')[:280]}…", flush=True)
        results.append({**test, "status": status, "evaluation": ev, "data": {
            "hits_count": len(get_hits(data)),
            "sources": ev["sources"],
            "warnings": [w.get("code") for w in (data.get("warnings") or [])],
        }})

    passed = sum(1 for r in results if r.get("status") == "PASS")
    failed = sum(1 for r in results if r.get("status") == "FAIL")
    errors = sum(1 for r in results if r.get("error"))
    print(f"\n{'='*72}\nSUMMARY: {passed} pass, {failed} fail, {errors} errors / {len(TESTS)} tests")
    return 1 if failed or errors else 0


if __name__ == "__main__":
    sys.exit(main())
