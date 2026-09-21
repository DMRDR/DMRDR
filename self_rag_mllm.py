from __future__ import annotations

import json
import re
from typing import Any, Callable

from config import settings
from lexical_graph import LexicalGraphRetriever, build_graph_from_corpus
from tools import parse_loose_json_object


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
    except Exception:
        return default
    return max(0.0, min(1.0, x))


def _to_choice_lines(choices: list[str] | None) -> str:
    out: list[str] = []
    for i, ch in enumerate(choices or []):
        out.append(f"{chr(ord('A') + i)}. {str(ch)}")
    return "\n".join(out)


def _question_with_choices(question: str, choices: list[str] | None) -> str:
    if not choices:
        return str(question or "").strip()
    return (
        f"{str(question or '').strip()}\n\n"
        f"选项:\n{_to_choice_lines(choices)}"
    ).strip()


def _dedupe_corpus(corpus: list[tuple[str, str]] | None) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for source, text in list(corpus or []):
        src = str(source or "unknown").strip() or "unknown"
        txt = str(text or "").strip()
        if not txt:
            continue
        key = (src, txt[:500])
        if key in seen:
            continue
        seen.add(key)
        out.append((src, txt))

    return out


def _compact_text(text: str, max_chars: int) -> str:
    s = str(text or "").strip()
    if max_chars <= 0:
        return ""
    return s if len(s) <= max_chars else s[:max_chars]


def _chat_json(
    llm: Any,
    messages: list[dict[str, Any]],
    fallback: Any,
    route: str = "reasoning",
    temperature: float = 0.0,
) -> Any:
    if llm is None:
        return fallback

    if hasattr(llm, "chat_json"):
        try:
            return llm.chat_json(
                messages,
                fallback=fallback,
                temperature=temperature,
                route=route,
            )
        except TypeError:
            return llm.chat_json(
                messages,
                fallback=fallback,
                route=route,
            )
        except Exception:
            return fallback

    try:
        raw = llm.chat(messages, temperature=temperature)
        return parse_loose_json_object(raw, fallback=fallback)
    except Exception:
        return fallback


def _format_docs_for_context(
    docs: list[dict[str, Any]],
    max_chars: int | None = None,
) -> str:
    max_chars = int(max_chars or getattr(settings, "self_rag_context_max_chars", 5000) or 5000)

    parts: list[str] = []
    used = 0

    for idx, d in enumerate(docs, start=1):
        source = str(d.get("source", "unknown")).strip() or "unknown"
        text = str(d.get("text", "")).strip()
        if not text:
            continue

        score = _safe_float(d.get("score", d.get("relevance", 0.0)), 0.0)
        block = f"[证据{idx} | source={source} | score={score:.3f}]\n{text[:1000]}"
        if used + len(block) > max_chars:
            remain = max(0, max_chars - used)
            if remain <= 100:
                break
            block = block[:remain]

        parts.append(block)
        used += len(block)

        if used >= max_chars:
            break

    return "\n\n".join(parts).strip()


def _retrieve_from_local_corpus(
    query: str,
    corpus: list[tuple[str, str]],
    top_k: int,
) -> list[dict[str, Any]]:
    if not corpus:
        return []

    try:
        index = build_graph_from_corpus(corpus)
        hits = LexicalGraphRetriever(index).retrieve(query, top_k=top_k)
    except Exception:
        return []

    docs: list[dict[str, Any]] = []
    for h in hits:
        docs.append(
            {
                "source": str(getattr(h, "source", "unknown")),
                "text": str(getattr(h, "text", "")),
                "score": _safe_float(getattr(h, "score", 0.0), 0.0),
                "chunk_id": str(getattr(h, "chunk_id", "")),
                "reasons": list(getattr(h, "reasons", []) or []),
            }
        )

    return docs


def _merge_docs(
    old_docs: list[dict[str, Any]],
    new_docs: list[dict[str, Any]],
    keep_k: int,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for d in list(old_docs or []) + list(new_docs or []):
        source = str(d.get("source", "unknown")).strip()
        text = str(d.get("text", "")).strip()
        if not text:
            continue
        key = (source, text[:300])
        if key in seen:
            continue
        seen.add(key)
        merged.append(dict(d))

    merged.sort(
        key=lambda x: (
            _safe_float(x.get("support", 0.0), 0.0),
            _safe_float(x.get("relevance", x.get("score", 0.0)), 0.0),
        ),
        reverse=True,
    )
    return merged[:keep_k]


def _predict_retrieval_delta(
    llm: Any,
    question: str,
    choices: list[str] | None,
    draft_answer: str,
    evidence_context: str,
    step_idx: int,
    mode: str,
) -> dict[str, Any]:
    mode = str(mode or "adaptive_retrieval").strip().lower()

    if mode == "always_retrieve":
        return {
            "delta_t": 1,
            "query": question,
            "reason": "SELF_RAG_MODE_ALWAYS_RETRIEVE",
            "confidence": 1.0,
        }

    if mode == "no_retrieval":
        return {
            "delta_t": 0,
            "query": "",
            "reason": "SELF_RAG_MODE_NO_RETRIEVAL",
            "confidence": 1.0,
        }

    fallback_delta = 1 if (step_idx == 0 and not evidence_context.strip()) else 0
    fallback = {
        "delta_t": fallback_delta,
        "query": question,
        "reason": "fallback_adaptive",
        "confidence": 0.5,
    }

    prompt = (
        "假设你是一个专业的 Self-RAG 反思控制器。"
        "请根据当前生成状态 x 判断是否需要检索。"
        "输出 JSON 对象："
        '{"delta_t":0 或 1,"query":"检索 query","reason":"原因","confidence":0.0 到 1.0}。'
        "规则："
        "1) 如果当前答案缺少证据支持、视觉/文本证据不足、选项之间仍无法区分，则 delta_t=1；"
        "2) 如果已有证据足以支持唯一答案，则 delta_t=0；"
        "3) query 必须短，并且只能用于本地候选证据检索，不要提出联网搜索。"
    )

    user = (
        f"问题:\n{_question_with_choices(question, choices)}\n\n"
        f"当前 step: {step_idx}\n"
        f"当前草稿答案:\n{draft_answer or '<EMPTY>'}\n\n"
        f"已有证据:\n{evidence_context or '<EMPTY>'}"
    )

    data = _chat_json(
        llm,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user},
        ],
        fallback=fallback,
        route="reasoning",
        temperature=0.0,
    )

    if not isinstance(data, dict):
        data = fallback

    delta = 1 if str(data.get("delta_t", fallback_delta)).strip() in {"1", "true", "True"} else 0
    query = str(data.get("query", "") or "").strip() or question
    return {
        "delta_t": delta,
        "query": query,
        "reason": str(data.get("reason", "") or "").strip(),
        "confidence": _clamp01(data.get("confidence", 0.5), default=0.5),
    }


def _assess_retrieved_docs(
    llm: Any,
    question: str,
    choices: list[str] | None,
    draft_answer: str,
    docs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not docs:
        return []

    payload = []
    for i, d in enumerate(docs):
        payload.append(
            {
                "idx": i,
                "source": str(d.get("source", "unknown")),
                "text": str(d.get("text", ""))[:1000],
            }
        )

    fallback = {
        "items": [
            {
                "idx": i,
                "relevance": _safe_float(d.get("score", 0.0), 0.0),
                "support": 0.0,
                "relevant": True,
                "supports_answer": False,
                "critique": "fallback",
            }
            for i, d in enumerate(docs)
        ]
    }

    prompt = (
        "假设你是一个专业的 Self-RAG 证据反思器。"
        "请判断每条候选证据是否与问题相关，以及是否支持当前答案。"
        "只输出 JSON 对象："
        '{"items":[{"idx":0,"relevant":true,"relevance":0.0-1.0,'
        '"supports_answer":true,"support":0.0-1.0,"critique":"..."}]}。'
        "不要引入候选证据以外的信息。"
    )

    user = json.dumps(
        {
            "question": _question_with_choices(question, choices),
            "draft_answer": draft_answer,
            "candidates": payload,
        },
        ensure_ascii=False,
    )

    data = _chat_json(
        llm,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user},
        ],
        fallback=fallback,
        route="verify",
        temperature=0.0,
    )

    if not isinstance(data, dict):
        data = fallback

    rows = data.get("items", [])
    if not isinstance(rows, list):
        rows = []

    score_by_idx: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        idx = row.get("idx")
        if not isinstance(idx, int) or not (0 <= idx < len(docs)):
            continue
        score_by_idx[idx] = row

    out: list[dict[str, Any]] = []
    for i, d in enumerate(docs):
        row = score_by_idx.get(i, {})
        item = dict(d)
        item["relevance"] = _clamp01(row.get("relevance", d.get("score", 0.0)), default=_safe_float(d.get("score", 0.0), 0.0))
        item["support"] = _clamp01(row.get("support", 0.0), default=0.0)
        item["relevant"] = bool(row.get("relevant", True))
        item["supports_answer"] = bool(row.get("supports_answer", False))
        item["critique"] = str(row.get("critique", "") or "").strip()
        out.append(item)

    out.sort(
        key=lambda x: (
            _safe_float(x.get("relevance", 0.0), 0.0),
            _safe_float(x.get("support", 0.0), 0.0),
        ),
        reverse=True,
    )
    return out


def _generate_answer(
    llm: Any,
    question: str,
    choices: list[str] | None,
    image_ref: str | None,
    context: str,
    answer_fn: Callable[..., str] | None,
    dataset_type: str,
) -> str:
    if callable(answer_fn):
        try:
            return str(
                answer_fn(
                    llm=llm,
                    question=question,
                    context=context,
                    image_ref=image_ref,
                    dataset_type=dataset_type,
                    choices=choices or [],
                )
            ).strip()
        except Exception:
            pass

    answer_constraint = ""
    if choices:
        answer_constraint = "这是选择题，最终答案必须严格等于某一个选项文本，不要输出解释。\n"

    prompt = (
        "假设你是一个专业多模态 VQA 或多跳推理答题器。"
        "请基于图像和给定本地证据回答问题。"
        "不要使用联网知识；若证据不足，仍给出最可能选项。\n"
        f"{answer_constraint}"
        f"问题:\n{_question_with_choices(question, choices)}\n\n"
        f"本地证据:\n{context or '<EMPTY>'}\n\n"
        "最终只输出答案。"
    )

    if llm is None:
        return ""

    try:
        if image_ref and hasattr(llm, "chat_with_image"):
            return str(
                llm.chat_with_image(
                    prompt=prompt,
                    image_url=image_ref,
                    temperature=0.1,
                    route="default",
                )
            ).strip()

        return str(
            llm.chat(
                [
                    {"role": "system", "content": "你是VQA评测助手。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                route="default",
            )
        ).strip()
    except TypeError:
        if image_ref and hasattr(llm, "chat_with_image"):
            return str(llm.chat_with_image(prompt=prompt, image_url=image_ref, temperature=0.1)).strip()
        return str(llm.chat([{"role": "user", "content": prompt}], temperature=0.1)).strip()
    except Exception:
        return ""


def _reflect_final_answer(
    llm: Any,
    question: str,
    choices: list[str] | None,
    answer: str,
    context: str,
) -> dict[str, Any]:
    fallback = {
        "answer": answer,
        "supported": bool(context.strip()),
        "confidence": 0.55 if context.strip() else 0.35,
        "needs_more_retrieval": False,
        "reason": "fallback",
    }

    prompt = (
        "假设你是一个专业的 Self-RAG 最终反思器。"
        "请判断最终答案是否被证据支持，是否还需要继续检索。"
        "只输出 JSON："
        '{"answer":"...","supported":true,"confidence":0.0-1.0,'
        '"needs_more_retrieval":false,"reason":"..."}。'
        "如果是选择题，answer 必须是某个选项文本。"
    )

    user = (
        f"问题:\n{_question_with_choices(question, choices)}\n\n"
        f"候选答案:\n{answer}\n\n"
        f"证据:\n{context or '<EMPTY>'}"
    )

    data = _chat_json(
        llm,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user},
        ],
        fallback=fallback,
        route="verify",
        temperature=0.0,
    )

    if not isinstance(data, dict):
        data = fallback

    return {
        "answer": str(data.get("answer", answer) or answer).strip(),
        "supported": bool(data.get("supported", fallback["supported"])),
        "confidence": _clamp01(data.get("confidence", fallback["confidence"]), default=fallback["confidence"]),
        "needs_more_retrieval": bool(data.get("needs_more_retrieval", False)),
        "reason": str(data.get("reason", "") or "").strip(),
    }


def _docs_to_evidence(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []

    for d in docs:
        rel = _clamp01(d.get("relevance", d.get("score", 0.0)), default=0.0)
        sup = _clamp01(d.get("support", 0.0), default=0.0)
        confidence = max(rel, sup)

        evidence.append(
            {
                "source": str(d.get("source", "self_rag_local")),
                "claim": "Self-RAG 本地检索证据",
                "excerpt": str(d.get("text", ""))[:1000],
                "confidence": confidence,
                "supports": [],
                "contradicts": [],
                "evidence_type": "kb_chunk",
                "premise_dependencies": [],
                "ambiguity": max(0.0, 1.0 - rel),
                "exclusivity": sup,
                "self_rag": {
                    "relevance": rel,
                    "support": sup,
                    "relevant": bool(d.get("relevant", True)),
                    "supports_answer": bool(d.get("supports_answer", False)),
                    "critique": str(d.get("critique", "")),
                    "chunk_id": str(d.get("chunk_id", "")),
                    "reasons": d.get("reasons", []),
                },
            }
        )

    return evidence


def run_self_rag_mllm(
    *,
    llm: Any,
    sample: dict[str, Any],
    image_ref: str | None,
    corpus: list[tuple[str, str]],
    answer_fn: Callable[..., str] | None = None,
    choices: list[str] | None = None,
    map_answer_fn: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """
    - 不实例化 SearchTool / WebpageTool；
    - 不调用 ToolRouter.run("search_web")；
    - 只使用传入的本地 corpus + image_ref + MLLM 反思。
    """
    q = str(sample.get("question", "") or "").strip()
    dataset_type = str(sample.get("dataset_type", "generic") or "generic").strip().lower()
    choices = choices if choices is not None else [str(x) for x in (sample.get("choices") or [])]
    corpus = _dedupe_corpus(corpus)

    mode = str(getattr(settings, "self_rag_mllm_mode", "adaptive_retrieval") or "adaptive_retrieval").strip().lower()
    max_steps = max(1, int(getattr(settings, "self_rag_max_steps", 3) or 3))
    top_k = max(1, int(getattr(settings, "self_rag_retrieval_top_k", 5) or 5))
    min_relevance = _clamp01(getattr(settings, "self_rag_min_relevance", 0.45), default=0.45)
    support_threshold = _clamp01(getattr(settings, "self_rag_support_threshold", 0.60), default=0.60)

    trace: dict[str, Any] = {
        "executor": "self_rag_mllm",
        "mode": mode,
        "no_web_search": True,
        "corpus_size": len(corpus),
        "steps": [],
    }

    if llm is None:
        return {
            "mode": "self_rag_mllm",
            "answer": "",
            "context": "",
            "evidence": [],
            "retrieval_trace": trace,
            "error": "mock_mode_no_llm",
            "final_confidence": 0.0,
            "decision_status": "fallback",
            "decision_reason": "mock_mode_no_llm",
        }

    kept_docs: list[dict[str, Any]] = []
    draft_answer = ""
    final_reflection: dict[str, Any] = {}
    last_query = q

    for step_idx in range(max_steps):
        evidence_context = _format_docs_for_context(kept_docs)
        delta = _predict_retrieval_delta(
            llm=llm,
            question=q,
            choices=choices,
            draft_answer=draft_answer,
            evidence_context=evidence_context,
            step_idx=step_idx,
            mode=mode,
        )

        retrieved_docs: list[dict[str, Any]] = []
        assessed_docs: list[dict[str, Any]] = []

        if int(delta.get("delta_t", 0)) == 1 and corpus:
            query = str(delta.get("query", "") or last_query or q).strip()
            retrieved_docs = _retrieve_from_local_corpus(
                query=query,
                corpus=corpus,
                top_k=top_k,
            )
            assessed_docs = _assess_retrieved_docs(
                llm=llm,
                question=q,
                choices=choices,
                draft_answer=draft_answer,
                docs=retrieved_docs,
            )
            useful_docs = [
                d for d in assessed_docs
                if bool(d.get("relevant", True))
                and _safe_float(d.get("relevance", 0.0), 0.0) >= min_relevance
            ]
            kept_docs = _merge_docs(
                kept_docs,
                useful_docs,
                keep_k=max(top_k * 2, top_k),
            )
            last_query = query

        context = _format_docs_for_context(kept_docs)
        draft_answer = _generate_answer(
            llm=llm,
            question=q,
            choices=choices,
            image_ref=image_ref,
            context=context,
            answer_fn=answer_fn,
            dataset_type=dataset_type,
        )

        final_reflection = _reflect_final_answer(
            llm=llm,
            question=q,
            choices=choices,
            answer=draft_answer,
            context=context,
        )

        trace["steps"].append(
            {
                "step": step_idx,
                "delta_t": int(delta.get("delta_t", 0)),
                "delta_reason": str(delta.get("reason", "")),
                "query": str(delta.get("query", "")),
                "retrieved_count": len(retrieved_docs),
                "kept_count": len(kept_docs),
                "draft_answer": draft_answer,
                "reflection": final_reflection,
            }
        )

        if (
            bool(final_reflection.get("supported", False))
            and not bool(final_reflection.get("needs_more_retrieval", False))
            and _safe_float(final_reflection.get("confidence", 0.0), 0.0) >= support_threshold
        ):
            break

        if not corpus:
            break

    answer = str(final_reflection.get("answer", draft_answer) or draft_answer).strip()
    if callable(map_answer_fn):
        try:
            answer = str(map_answer_fn(answer)).strip()
        except Exception:
            pass

    final_confidence = _clamp01(final_reflection.get("confidence", 0.0), default=0.0)
    supported = bool(final_reflection.get("supported", False))

    decision_status = "finalized" if supported and final_confidence >= support_threshold else "fallback"
    decision_reason = (
        "self_rag_supported"
        if decision_status == "finalized"
        else str(final_reflection.get("reason", "self_rag_low_support") or "self_rag_low_support")
    )

    context = _format_docs_for_context(kept_docs)
    evidence = _docs_to_evidence(kept_docs)

    return {
        "mode": "self_rag_mllm",
        "answer": answer,
        "context": context,
        "evidence": evidence,
        "tool_logs": [],
        "retrieval_trace": trace,
        "kg_trace": {},
        "raw_hits": [],
        "error": None,
        "final_confidence": final_confidence,
        "decision_status": decision_status,
        "decision_reason": decision_reason,
    }