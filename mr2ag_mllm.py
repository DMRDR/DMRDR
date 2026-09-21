from __future__ import annotations

import json
import math
import re
from typing import Any, Callable

from config import settings
from embeddings import EmbeddingReranker


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
    except Exception:
        return default
    if math.isnan(x):
        return default
    return max(0.0, min(1.0, x))


def _safe_int(value: Any, default: int = 1, low: int = 1, high: int = 20) -> int:
    try:
        x = int(value)
    except Exception:
        x = default
    return max(low, min(high, x))


def _parse_jsonish(text: Any, fallback: Any) -> Any:
    raw = str(text or "").strip()
    if not raw:
        return fallback

    try:
        return json.loads(raw)
    except Exception:
        pass

    fenced = re.findall(r"```(?:json)?\s*(.*?)```", raw, flags=re.DOTALL | re.IGNORECASE)
    for block in fenced:
        try:
            return json.loads(block.strip())
        except Exception:
            pass

    decoder = json.JSONDecoder()
    for i, ch in enumerate(raw):
        if ch not in "{[":
            continue
        try:
            obj, _ = decoder.raw_decode(raw[i:])
            return obj
        except Exception:
            continue

    return fallback


def _question_terms(text: str) -> set[str]:
    toks = re.findall(r"[A-Za-z0-9_\-\u4e00-\u9fff]+", str(text or "").lower())
    stop = {
        "what", "which", "where", "when", "why", "how", "the", "and", "for", "with",
        "image", "picture", "photo", "question", "answer", "option", "choice",
        "什么", "哪个", "哪里", "为什么", "如何", "图片", "图像", "问题", "答案", "选项",
    }
    return {t for t in toks if len(t) >= 2 and t not in stop}


def _lexical_score(question: str, text: str) -> float:
    q_terms = _question_terms(question)
    if not q_terms:
        return 0.0
    low = str(text or "").lower()
    overlap = sum(1 for t in q_terms if t in low)
    return max(0.0, min(1.0, overlap / max(len(q_terms), 1)))


def _choices_block(choices: list[str] | None) -> str:
    if not choices:
        return "(no choices)"
    return "\n".join([f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(choices[:10])])


def _normalize_corpus(corpus: list[tuple[str, str]] | None) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for source, text in list(corpus or []):
        src = str(source or "unknown").strip()
        txt = str(text or "").strip()
        if not txt:
            continue
        key = (src, txt[:300])
        if key in seen:
            continue
        seen.add(key)
        docs.append({"source": src, "text": txt})

    return docs


def _build_context(docs: list[dict[str, Any]], max_chars: int) -> str:
    parts: list[str] = []
    used = 0

    for i, d in enumerate(docs, start=1):
        src = str(d.get("source", "unknown"))
        txt = str(d.get("text", "")).strip()
        if not txt:
            continue

        row = f"[{i}] source={src}\n{txt[:1200]}"
        if used + len(row) > max_chars:
            remain = max(0, max_chars - used)
            if remain > 120:
                parts.append(row[:remain])
            break

        parts.append(row)
        used += len(row)

    return "\n\n".join(parts).strip()


def _call_retrieval_reflection(
    *,
    llm: Any,
    question: str,
    image_ref: str | None,
    choices: list[str],
    corpus_size: int,
) -> dict[str, Any]:
    mode = str(getattr(settings, "mr2ag_mllm_mode", "auto") or "auto").strip().lower()
    max_depth = int(getattr(settings, "mr2ag_max_retrieval_depth", 5) or 5)
    default_top_k = int(getattr(settings, "mr2ag_retrieval_top_k", 5) or 5)

    if mode == "no_retrieval":
        return {
            "need_retrieval": False,
            "retrieval_depth": 0,
            "confidence": 1.0,
            "reason": "forced_no_retrieval",
            "mode": mode,
        }

    if mode == "always_retrieve":
        return {
            "need_retrieval": corpus_size > 0,
            "retrieval_depth": min(default_top_k, max_depth, max(corpus_size, 1)),
            "confidence": 1.0,
            "reason": "forced_always_retrieve",
            "mode": mode,
        }

    prompt = (
        "你正在执行 mR^2AG 的 Retrieval-Reflection。\n"
        "任务：判断当前多模态 VQA 问题是否需要外部候选证据检索，以及需要检索多少条候选证据。\n"
        "只输出 JSON 对象，不要输出解释性文字。\n"
        "JSON schema:\n"
        "{\n"
        '  "need_retrieval": true,\n'
        '  "retrieval_depth": 3,\n'
        '  "confidence": 0.0,\n'
        '  "reason": "..."\n'
        "}\n"
        "判断原则：\n"
        "1) 如果问题主要依赖图像可见事实、颜色、数量、位置、OCR，可倾向 no retrieval；\n"
        "2) 如果问题需要百科、外部常识、科学概念、实体属性、背景知识，可倾向 retrieval；\n"
        "3) retrieval_depth 在 1 到 max_depth 之间，证据越可能冗余，depth 越小。\n\n"
        f"max_depth: {max_depth}\n"
        f"candidate_corpus_size: {corpus_size}\n"
        f"question:\n{question}\n\n"
        f"choices:\n{_choices_block(choices)}"
    )

    fallback = {
        "need_retrieval": corpus_size > 0,
        "retrieval_depth": min(default_top_k, max_depth, max(corpus_size, 1)),
        "confidence": 0.5,
        "reason": "fallback_reflection",
    }

    if llm is None:
        return fallback

    try:
        if image_ref:
            raw = llm.chat_with_image(
                prompt=prompt,
                image_url=image_ref,
                temperature=0.0,
                route="vision",
            )
            data = _parse_jsonish(raw, fallback)
        else:
            data = llm.chat_json(
                [
                    {"role": "system", "content": "你是 mR2AG Retrieval-Reflection 判断器，只输出 JSON。"},
                    {"role": "user", "content": prompt},
                ],
                fallback=fallback,
                temperature=0.0,
                route="reasoning",
            )
    except Exception:
        data = fallback

    if not isinstance(data, dict):
        data = fallback

    conf = _safe_float(data.get("confidence", 0.5), 0.5)
    threshold = float(getattr(settings, "mr2ag_retrieval_reflection_threshold", 0.50) or 0.50)
    need = bool(data.get("need_retrieval", False)) and corpus_size > 0 and conf >= threshold
    depth = _safe_int(
        data.get("retrieval_depth", default_top_k),
        default=default_top_k,
        low=1,
        high=max(1, min(max_depth, max(corpus_size, 1))),
    )

    return {
        "need_retrieval": need,
        "retrieval_depth": depth if need else 0,
        "confidence": conf,
        "reason": str(data.get("reason", "") or "").strip(),
        "mode": mode,
        "raw": data,
    }


def _retrieve_local_candidates(
    *,
    question: str,
    corpus: list[tuple[str, str]] | None,
    top_k: int,
) -> list[dict[str, Any]]:
    docs = _normalize_corpus(corpus)
    if not docs:
        return []

    for d in docs:
        d["lexical_score"] = _lexical_score(question, f"{d.get('source', '')}\n{d.get('text', '')}")
        d["retrieval_score"] = float(d["lexical_score"])

    try:
        reranker = EmbeddingReranker()
        ranked = reranker.rerank_docs(question, docs, top_k=max(top_k, 1))
        out: list[dict[str, Any]] = []
        for row in ranked:
            item = dict(row)
            item["retrieval_score"] = _safe_float(
                item.get("rerank_score", item.get("retrieval_score", item.get("lexical_score", 0.0))),
                0.0,
            )
            out.append(item)
        return out[:top_k]
    except Exception:
        docs.sort(key=lambda x: float(x.get("retrieval_score", 0.0)), reverse=True)
        return docs[:top_k]


def _call_relevance_reflection(
    *,
    llm: Any,
    question: str,
    image_ref: str | None,
    choices: list[str],
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not candidates:
        return []

    min_relevance = float(getattr(settings, "mr2ag_min_relevance", 0.45) or 0.45)

    payload = []
    for i, d in enumerate(candidates):
        payload.append(
            {
                "idx": i,
                "source": str(d.get("source", "unknown")),
                "retrieval_score": _safe_float(d.get("retrieval_score", 0.0), 0.0),
                "text": str(d.get("text", ""))[:1400],
            }
        )

    prompt = (
        "假设你正在执行 mR^2AG 的 Relevance-Reflection。\n"
        "任务：判断每条候选证据是否真正支持当前 VQA 问题，并定位最有用的证据片段。\n"
        "只输出 JSON 数组，不要输出解释性文字。\n"
        "每个元素格式：\n"
        "{\n"
        '  "idx": 0,\n'
        '  "relevant": true,\n'
        '  "relevance_score": 0.0,\n'
        '  "evidence_span": "...",\n'
        '  "candidate_answer": "...",\n'
        '  "answer_confidence": 0.0,\n'
        '  "reason": "..."\n'
        "}\n"
        "要求：\n"
        "1) evidence_span 必须来自候选证据文本或对其高度忠实的压缩表达；\n"
        "2) 如果证据不能支持答案，relevant=false；\n"
        "3) 如果是选择题，candidate_answer 优先输出选项字母或选项文本；\n"
        "4) 不要使用 web search，不要编造候选证据之外的信息。\n\n"
        f"question:\n{question}\n\n"
        f"choices:\n{_choices_block(choices)}\n\n"
        f"candidates:\n{json.dumps(payload, ensure_ascii=False)}"
    )

    fallback: list[dict[str, Any]] = []
    for d in candidates:
        score = _safe_float(d.get("retrieval_score", 0.0), 0.0)
        fallback.append(
            {
                "idx": len(fallback),
                "relevant": score >= min_relevance,
                "relevance_score": score,
                "evidence_span": str(d.get("text", ""))[:500],
                "candidate_answer": "",
                "answer_confidence": 0.0,
                "reason": "fallback_relevance_by_retrieval_score",
            }
        )

    if llm is None:
        return fallback

    try:
        if image_ref:
            raw = llm.chat_with_image(
                prompt=prompt,
                image_url=image_ref,
                temperature=0.0,
                route="vision",
            )
            data = _parse_jsonish(raw, fallback)
        else:
            data = llm.chat_json(
                [
                    {"role": "system", "content": "假设你是一个专业的 mR^2AG Relevance-Reflection 判断器，只输出 JSON。"},
                    {"role": "user", "content": prompt},
                ],
                fallback=fallback,
                temperature=0.0,
                route="reasoning",
            )
    except Exception:
        data = fallback

    if not isinstance(data, list):
        data = fallback

    normalized: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue

        idx = item.get("idx")
        if not isinstance(idx, int) or not (0 <= idx < len(candidates)):
            continue

        source_doc = candidates[idx]
        rel_score = _safe_float(item.get("relevance_score", 0.0), 0.0)
        ans_conf = _safe_float(item.get("answer_confidence", 0.0), 0.0)
        relevant = bool(item.get("relevant", False)) and rel_score >= min_relevance

        normalized.append(
            {
                "idx": idx,
                "source": str(source_doc.get("source", "unknown")),
                "text": str(source_doc.get("text", "")),
                "retrieval_score": _safe_float(source_doc.get("retrieval_score", 0.0), 0.0),
                "relevant": relevant,
                "relevance_score": rel_score,
                "evidence_span": str(item.get("evidence_span", "") or "").strip()[:1000],
                "candidate_answer": str(item.get("candidate_answer", "") or "").strip(),
                "answer_confidence": ans_conf,
                "reason": str(item.get("reason", "") or "").strip(),
            }
        )

    if not normalized:
        normalized = fallback

    normalized.sort(
        key=lambda x: (
            _safe_float(x.get("retrieval_score", 0.0), 0.0)
            * max(_safe_float(x.get("relevance_score", 0.0), 0.0), 0.01)
            * max(_safe_float(x.get("answer_confidence", 0.0), 0.0), 0.25)
        ),
        reverse=True,
    )
    return normalized


def _evidence_from_reflections(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for row in rows:
        if not bool(row.get("relevant", False)):
            continue

        retrieval_score = _safe_float(row.get("retrieval_score", 0.0), 0.0)
        relevance_score = _safe_float(row.get("relevance_score", 0.0), 0.0)
        answer_conf = _safe_float(row.get("answer_confidence", 0.0), 0.0)

        confidence = max(0.0, min(1.0, 0.35 * retrieval_score + 0.45 * relevance_score + 0.20 * answer_conf))

        evidence.append(
            {
                "source": str(row.get("source", "unknown")),
                "claim": "mR2AG Relevance-Reflection selected evidence",
                "excerpt": str(row.get("evidence_span", "") or row.get("text", ""))[:1000],
                "confidence": confidence,
                "supports": [str(row.get("candidate_answer", "")).strip()] if str(row.get("candidate_answer", "")).strip() else [],
                "contradicts": [],
                "evidence_type": "mr2ag_reflected_evidence",
                "premise_dependencies": [],
                "ambiguity": max(0.0, min(1.0, 1.0 - relevance_score)),
                "exclusivity": relevance_score,
            }
        )

    return evidence


def run_mr2ag_mllm(
    *,
    llm: Any,
    sample: dict[str, Any],
    image_ref: str | None,
    corpus: list[tuple[str, str]] | None,
    answer_fn: Callable[..., str],
    choices: list[str] | None = None,
    map_answer_fn: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    question = str(sample.get("question", "") or "").strip()
    dataset_type = str(sample.get("dataset_type", "generic") or "generic").strip().lower()
    choices = choices or []

    trace: dict[str, Any] = {
        "executor": "mr2ag_mllm",
        "no_web_search": bool(getattr(settings, "mr2ag_no_web_search", True)),
        "retrieval_reflection": {},
        "retrieved_count": 0,
        "relevance_reflection_count": 0,
        "selected_evidence_count": 0,
    }

    tool_logs: list[dict[str, Any]] = []

    docs = _normalize_corpus(corpus)
    reflection = _call_retrieval_reflection(
        llm=llm,
        question=question,
        image_ref=image_ref,
        choices=choices,
        corpus_size=len(docs),
    )
    trace["retrieval_reflection"] = reflection
    tool_logs.append(
        {
            "tool_name": "mr2ag_retrieval_reflection",
            "input_data": {
                "question": question,
                "has_image": bool(image_ref),
                "candidate_corpus_size": len(docs),
            },
            "output_data": json.dumps(reflection, ensure_ascii=False),
        }
    )

    # mR2AG w/o Retrieval：直接让 MLLM 依靠图像和题目回答。
    if not bool(reflection.get("need_retrieval", False)):
        try:
            answer = answer_fn(
                llm=llm,
                question=question,
                context="",
                image_ref=image_ref,
                dataset_type=dataset_type,
                choices=choices,
            )
            answer = str(answer or "").strip()
            if map_answer_fn is not None:
                answer = str(map_answer_fn(answer))
            conf = max(0.50, _safe_float(reflection.get("confidence", 0.50), 0.50))
            return {
                "mode": "mr2ag_mllm",
                "answer": answer,
                "context": "",
                "evidence": [],
                "tool_logs": tool_logs,
                "retrieval_trace": trace,
                "kg_trace": {},
                "raw_hits": [],
                "error": None,
                "final_confidence": conf,
                "decision_status": "finalized" if answer else "fallback",
                "decision_reason": "mr2ag_no_retrieval",
            }
        except Exception as exc:
            return {
                "mode": "mr2ag_mllm",
                "answer": "",
                "context": "",
                "evidence": [],
                "tool_logs": tool_logs,
                "retrieval_trace": trace,
                "kg_trace": {},
                "raw_hits": [],
                "error": f"{type(exc).__name__}: {exc}",
                "final_confidence": 0.0,
                "decision_status": "fallback",
                "decision_reason": "mr2ag_no_retrieval_answer_exception",
            }

    depth = _safe_int(
        reflection.get("retrieval_depth", getattr(settings, "mr2ag_retrieval_top_k", 5)),
        default=int(getattr(settings, "mr2ag_retrieval_top_k", 5) or 5),
        low=1,
        high=max(1, min(len(docs), int(getattr(settings, "mr2ag_max_retrieval_depth", 5) or 5))),
    )

    candidates = _retrieve_local_candidates(
        question=question,
        corpus=corpus,
        top_k=depth,
    )
    trace["retrieved_count"] = len(candidates)
    tool_logs.append(
        {
            "tool_name": "mr2ag_local_retrieve",
            "input_data": {"top_k": depth, "no_web_search": True},
            "output_data": json.dumps(
                [
                    {
                        "source": d.get("source", "unknown"),
                        "retrieval_score": _safe_float(d.get("retrieval_score", 0.0), 0.0),
                        "text": str(d.get("text", ""))[:300],
                    }
                    for d in candidates
                ],
                ensure_ascii=False,
            ),
        }
    )

    relevance_rows = _call_relevance_reflection(
        llm=llm,
        question=question,
        image_ref=image_ref,
        choices=choices,
        candidates=candidates,
    )
    trace["relevance_reflection_count"] = len(relevance_rows)
    tool_logs.append(
        {
            "tool_name": "mr2ag_relevance_reflection",
            "input_data": {"candidate_count": len(candidates), "has_image": bool(image_ref)},
            "output_data": json.dumps(
                [
                    {
                        "idx": r.get("idx"),
                        "source": r.get("source"),
                        "relevant": r.get("relevant"),
                        "relevance_score": r.get("relevance_score"),
                        "candidate_answer": r.get("candidate_answer"),
                        "answer_confidence": r.get("answer_confidence"),
                        "reason": r.get("reason"),
                    }
                    for r in relevance_rows
                ],
                ensure_ascii=False,
            ),
        }
    )

    selected_rows = [r for r in relevance_rows if bool(r.get("relevant", False))]
    if not selected_rows and candidates:
        selected_rows = relevance_rows[:1]

    evidence = _evidence_from_reflections(selected_rows)
    trace["selected_evidence_count"] = len(evidence)

    max_chars = int(getattr(settings, "mr2ag_context_max_chars", 5000) or 5000)
    context_docs = []
    for row in selected_rows:
        context_docs.append(
            {
                "source": row.get("source", "unknown"),
                "text": row.get("evidence_span") or row.get("text", ""),
                "retrieval_score": row.get("retrieval_score", 0.0),
                "relevance_score": row.get("relevance_score", 0.0),
            }
        )
    context = _build_context(context_docs, max_chars=max_chars)

    best_reflected_answer = ""
    best_score = 0.0
    for row in selected_rows:
        score = (
            _safe_float(row.get("retrieval_score", 0.0), 0.0)
            * max(_safe_float(row.get("relevance_score", 0.0), 0.0), 0.01)
            * max(_safe_float(row.get("answer_confidence", 0.0), 0.0), 0.25)
        )
        if score > best_score:
            best_score = score
            best_reflected_answer = str(row.get("candidate_answer", "") or "").strip()

    try:
        answer = answer_fn(
            llm=llm,
            question=question,
            context=context,
            image_ref=image_ref,
            dataset_type=dataset_type,
            choices=choices,
        )
        answer = str(answer or "").strip()
    except Exception:
        answer = ""

    if not answer:
        answer = best_reflected_answer

    if map_answer_fn is not None:
        answer = str(map_answer_fn(answer))

    confidence = max(
        best_score,
        max([_safe_float(e.get("confidence", 0.0), 0.0) for e in evidence], default=0.0),
        0.0,
    )

    threshold = float(getattr(settings, "mr2ag_answer_confidence_threshold", 0.45) or 0.45)
    decision_status = "finalized" if answer and confidence >= threshold else "fallback"
    decision_reason = "mr2ag_retrieval_relevance_reflection" if decision_status == "finalized" else "mr2ag_low_confidence_or_empty_answer"

    return {
        "mode": "mr2ag_mllm",
        "answer": answer,
        "context": context,
        "evidence": evidence,
        "tool_logs": tool_logs,
        "retrieval_trace": trace,
        "kg_trace": {},
        "raw_hits": candidates,
        "error": None,
        "final_confidence": confidence,
        "decision_status": decision_status,
        "decision_reason": decision_reason,
    }