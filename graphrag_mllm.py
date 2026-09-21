from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from config import settings
from lexical_graph import LexicalGraphRetriever, build_graph_from_corpus


def _clip_text(text: Any, max_chars: int) -> str:
    s = str(text or "").strip()
    if max_chars <= 0:
        return ""
    return s if len(s) <= max_chars else s[:max_chars] + "\n...<truncated>"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
    except Exception:
        return default
    return max(0.0, min(1.0, x))


def _choices_block(choices: list[str] | None) -> str:
    choices = choices or []
    lines: list[str] = []
    for i, ch in enumerate(choices[:10]):
        lines.append(f"{chr(ord('A') + i)}. {ch}")
    return "\n".join(lines)


def _dedupe_corpus(
    corpus: list[tuple[str, str]] | None,
    extra_docs: list[dict[str, str]] | None = None,
) -> list[tuple[str, str]]:
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

    for d in list(extra_docs or []):
        if not isinstance(d, dict):
            continue
        src = str(d.get("source", "unknown")).strip() or "unknown"
        txt = str(d.get("text", "")).strip()
        if not txt:
            continue
        key = (src, txt[:500])
        if key in seen:
            continue
        seen.add(key)
        out.append((src, txt))

    return out


def _visual_textualize(
    *,
    llm: Any,
    image_ref: str | None,
    question: str,
    choices: list[str] | None,
) -> tuple[str, dict[str, Any]]:
    trace: dict[str, Any] = {
        "called": False,
        "ok": False,
        "error": "",
        "route": "vision",
        "max_chars": int(getattr(settings, "graphrag_visual_summary_max_chars", 1600) or 1600),
    }

    if llm is None or not str(image_ref or "").strip():
        trace["error"] = "no_llm_or_no_image"
        return "", trace

    option_text = _choices_block(choices)
    prompt = (
        "假设你正在为 GraphRAG 检索生成图像的客观文本化描述。"
        "请只描述图像中可直接观察到的事实，不要直接回答题目，不要猜测不可见知识。\n"
        "需要覆盖：对象、属性、数量、空间关系、动作、场景、可读文字/OCR、可能影响选项判断的视觉细节。\n"
        "如果存在不确定性，请明确写出不确定点。\n\n"
        f"题目:\n{question}\n\n"
        f"选项:\n{option_text if option_text else '(无选项)'}\n\n"
        "输出格式：一段紧凑的客观中文描述。"
    )

    try:
        try:
            raw = llm.chat_with_image(
                prompt=prompt,
                image_url=image_ref,
                temperature=0.0,
                route="vision",
            )
        except TypeError:
            raw = llm.chat_with_image(
                prompt=prompt,
                image_url=image_ref,
                temperature=0.0,
            )

        text = _clip_text(
            raw,
            int(getattr(settings, "graphrag_visual_summary_max_chars", 1600) or 1600),
        )
        trace["ok"] = bool(text)
        trace["called"] = True
        return text, trace

    except Exception as exc:
        trace["called"] = True
        trace["error"] = f"{type(exc).__name__}: {exc}"
        return "", trace


def _build_graphrag_query(
    *,
    question: str,
    choices: list[str] | None,
    visual_summary: str,
) -> str:
    option_text = _choices_block(choices)

    parts = [
        "请根据问题、候选选项和图像客观描述检索结构化证据。",
        "重点检索能支持或排除候选答案的实体、关系、上下文片段、社区摘要或相关节点。",
        "",
        f"问题:\n{question}",
    ]

    if option_text:
        parts.extend(["", f"选项:\n{option_text}"])

    if visual_summary:
        parts.extend(["", f"图像客观描述:\n{visual_summary}"])

    return _clip_text("\n".join(parts), 2400)


def _resolve_graphrag_methods() -> list[str]:
    mode = str(getattr(settings, "graphrag_mllm_mode", "local_global") or "local_global").strip().lower()
    allowed = {"local", "global", "drift", "basic"}

    if mode in allowed:
        return [mode]

    if mode == "proxy_only":
        return []

    methods = []
    for m in list(getattr(settings, "graphrag_query_methods", []) or []):
        mm = str(m).strip().lower()
        if mm in allowed and mm not in methods:
            methods.append(mm)

    return methods or ["local", "global"]


def _find_graphrag_cli() -> str:
    configured = str(getattr(settings, "graphrag_cli_bin", "graphrag") or "graphrag").strip()
    if not configured:
        configured = "graphrag"

    if Path(configured).expanduser().is_file():
        return str(Path(configured).expanduser().resolve())

    found = shutil.which(configured)
    return found or ""


def _run_graphrag_cli_query(
    *,
    query: str,
    method: str,
) -> tuple[str, dict[str, Any]]:
    conda_env = str(
        getattr(settings, "graphrag_conda_env", "")
        or os.getenv("GRAPHRAG_CONDA_ENV", "")
    ).strip()

    conda_exe = str(
        getattr(settings, "graphrag_conda_exe", "")
        or os.getenv("GRAPHRAG_CONDA_EXE", "")
        or os.getenv("CONDA_EXE", "")
        or "conda"
    ).strip()

    configured_cli = str(
        getattr(settings, "graphrag_cli_bin", "graphrag")
        or "graphrag"
    ).strip() or "graphrag"

    trace: dict[str, Any] = {
        "provider": "microsoft_graphrag_cli",
        "method": method,
        "ok": False,
        "error": "",
        "returncode": None,
        "root": str(getattr(settings, "graphrag_root", "") or ""),
        "data_dir": str(getattr(settings, "graphrag_data_dir", "") or ""),
        "command_mode": "conda_run" if conda_env else "direct_cli",
        "graphrag_cli_bin": configured_cli,
        "graphrag_conda_env": conda_env,
        "graphrag_conda_exe": conda_exe if conda_env else "",
    }

    root = str(getattr(settings, "graphrag_root", "") or "").strip()
    data_dir = str(getattr(settings, "graphrag_data_dir", "") or "").strip()

    if not root:
        trace["error"] = "missing_GRAPHRAG_ROOT"
        return "", trace

    root_path = Path(root).expanduser()
    if not root_path.exists():
        trace["error"] = f"GRAPHRAG_ROOT_not_found:{root_path}"
        return "", trace

    if conda_env:
        if Path(conda_exe).expanduser().is_file():
            resolved_conda = str(Path(conda_exe).expanduser().resolve())
        else:
            resolved_conda = shutil.which(conda_exe) or ""

        if not resolved_conda:
            trace["error"] = "conda_exe_not_found_for_GRAPHRAG_CONDA_ENV"
            return "", trace

        cli = configured_cli
    else:
        cli = _find_graphrag_cli()
        if not cli:
            trace["error"] = "graphrag_cli_not_found"
            return "", trace

    inner_cmd = [
        cli,
        "query",
        "--root",
        str(root_path),
        "--method",
        method,
        "--response-type",
        "List concise evidence bullets with entity, relation, source hint, and uncertainty. Do not answer directly.",
    ]

    if data_dir:
        inner_cmd.extend(["--data", str(Path(data_dir).expanduser())])

    inner_cmd.append(query)

    if conda_env:
        cmd = [
            resolved_conda,
            "run",
            "-n",
            conda_env,
            *inner_cmd,
        ]
    else:
        cmd = inner_cmd

    trace["cmd_preview"] = " ".join(str(x) for x in cmd[:16])

    env = os.environ.copy()
    env["GRAPHRAG_NO_WEB_SEARCH"] = "1"
    env.setdefault("PYTHONNOUSERSITE", "1")

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=int(getattr(settings, "graphrag_query_timeout_s", 90) or 90),
            env=env,
        )
    except Exception as exc:
        trace["error"] = f"{type(exc).__name__}: {exc}"
        return "", trace

    trace["returncode"] = int(proc.returncode)
    stdout = str(proc.stdout or "").strip()
    stderr = str(proc.stderr or "").strip()

    if proc.returncode != 0:
        trace["error"] = _clip_text(stderr or stdout, 1200)
        return "", trace

    cleaned = _clean_graphrag_output(stdout)
    trace["ok"] = bool(cleaned)
    if not cleaned:
        trace["error"] = _clip_text(stderr or "empty_graphrag_output", 1200)

    return cleaned, trace


def _clean_graphrag_output(text: str) -> str:
    s = str(text or "").strip()
    if not s:
        return ""

    # 去掉一些常见 CLI 日志行，保留最终文本。
    lines = []
    for line in s.splitlines():
        raw = line.rstrip()
        low = raw.strip().lower()
        if not raw.strip():
            continue
        if low.startswith(("info:", "warning:", "success:", "creating", "running")):
            continue
        if re.match(r"^\[\d{4}-\d{2}-\d{2}", raw):
            continue
        lines.append(raw)

    return "\n".join(lines).strip() or s


def _proxy_graph_retrieve(
    *,
    query: str,
    corpus: list[tuple[str, str]],
    visual_summary: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    docs = _dedupe_corpus(
        corpus,
        extra_docs=(
            [{"source": "visual_textualization", "text": visual_summary}]
            if visual_summary else []
        ),
    )

    trace: dict[str, Any] = {
        "provider": "lexical_graph_fallback",
        "ok": False,
        "doc_count": len(docs),
        "top_k": int(getattr(settings, "graphrag_retrieval_top_k", 8) or 8),
        "error": "",
    }

    if not docs:
        trace["error"] = "empty_local_corpus"
        return [], trace

    try:
        index = build_graph_from_corpus(docs)
        retriever = LexicalGraphRetriever(index)
        hits = retriever.retrieve(
            query,
            top_k=int(getattr(settings, "graphrag_retrieval_top_k", 8) or 8),
        )
    except Exception as exc:
        trace["error"] = f"{type(exc).__name__}: {exc}"
        return [], trace

    rows: list[dict[str, Any]] = []
    for h in hits:
        rows.append(
            {
                "provider": "lexical_graph_fallback",
                "method": "entity_traversal",
                "source": str(getattr(h, "source", "")),
                "text": str(getattr(h, "text", "")),
                "score": float(getattr(h, "score", 0.0) or 0.0),
                "reasons": list(getattr(h, "reasons", []) or []),
            }
        )

    trace["ok"] = bool(rows)
    trace["hit_count"] = len(rows)
    return rows, trace


def _build_context(
    *,
    question: str,
    visual_summary: str,
    official_outputs: list[dict[str, Any]],
    proxy_hits: list[dict[str, Any]],
) -> str:
    parts: list[str] = []

    if visual_summary:
        parts.append("【视觉文本化描述 \\hat{c}】\n" + _clip_text(visual_summary, 1800))

    if official_outputs:
        lines = []
        for item in official_outputs:
            method = str(item.get("method", "unknown"))
            text = _clip_text(item.get("text", ""), 1800)
            if text:
                lines.append(f"[GraphRAG:{method}]\n{text}")
        if lines:
            parts.append("【GraphRAG 结构化证据 \\mathcal{G}】\n" + "\n\n".join(lines))

    if proxy_hits:
        lines = []
        for i, hit in enumerate(proxy_hits[: int(getattr(settings, "graphrag_retrieval_top_k", 8) or 8)], start=1):
            src = str(hit.get("source", "unknown"))
            score = hit.get("score", 0.0)
            text = _clip_text(hit.get("text", ""), 700)
            if text:
                lines.append(f"[ProxyGraph:{i} source={src} score={score}]\n{text}")
        if lines:
            parts.append("【本地代理图证据 fallback】\n" + "\n\n".join(lines))

    if not parts:
        parts.append("【GraphRAG 结构化证据】\n未检索到可用结构化证据。")

    context = "\n\n".join(parts)
    return _clip_text(
        context,
        int(getattr(settings, "graphrag_context_max_chars", 6000) or 6000),
    )


def _evidence_from_outputs(
    *,
    visual_summary: str,
    official_outputs: list[dict[str, Any]],
    proxy_hits: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []

    if visual_summary:
        evidence.append(
            {
                "source": "visual_textualization",
                "claim": "图像客观文本化描述",
                "excerpt": visual_summary,
                "confidence": 0.72,
                "supports": [],
                "contradicts": [],
                "evidence_type": "image_fact",
                "premise_dependencies": [],
                "ambiguity": 0.18,
                "exclusivity": 0.32,
            }
        )

    for item in official_outputs:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        method = str(item.get("method", "unknown"))
        evidence.append(
            {
                "source": f"graphrag:{method}",
                "claim": "GraphRAG 结构化检索证据",
                "excerpt": _clip_text(text, 1400),
                "confidence": 0.76 if method == "local" else 0.70,
                "supports": [],
                "contradicts": [],
                "evidence_type": "graph",
                "premise_dependencies": [],
                "ambiguity": 0.22,
                "exclusivity": 0.42,
            }
        )

    for hit in proxy_hits:
        text = str(hit.get("text", "")).strip()
        if not text:
            continue
        evidence.append(
            {
                "source": f"proxy_graph:{hit.get('source', 'unknown')}",
                "claim": "本地代理图检索证据",
                "excerpt": _clip_text(text, 900),
                "confidence": 0.60,
                "supports": [],
                "contradicts": [],
                "evidence_type": "graph",
                "premise_dependencies": ["proxy_graph_fallback_not_official_graphrag"],
                "ambiguity": 0.32,
                "exclusivity": 0.30,
            }
        )

    return evidence[:10]


def run_graphrag_mllm(
    *,
    llm: Any,
    sample: dict[str, Any],
    image_ref: str | None,
    corpus: list[tuple[str, str]],
    answer_fn: Any,
    choices: list[str] | None = None,
    map_answer_fn: Any | None = None,
) -> dict[str, Any]:
    question = str(sample.get("question", "")).strip()
    dataset_type = str(sample.get("dataset_type", "generic")).strip().lower() or "generic"
    choices = choices or []

    trace: dict[str, Any] = {
        "executor": "graphrag_mllm",
        "no_web_search": True,
        "official_graphrag": {
            "enabled": str(getattr(settings, "graphrag_mllm_mode", "local_global")).lower() != "proxy_only",
            "root": str(getattr(settings, "graphrag_root", "") or ""),
            "methods": [],
            "results": [],
        },
        "proxy_fallback": {},
    }

    visual_summary, visual_trace = _visual_textualize(
        llm=llm,
        image_ref=image_ref,
        question=question,
        choices=choices,
    )
    trace["visual_textualization"] = visual_trace

    query = _build_graphrag_query(
        question=question,
        choices=choices,
        visual_summary=visual_summary,
    )
    trace["query"] = _clip_text(query, 1200)

    official_outputs: list[dict[str, Any]] = []
    official_traces: list[dict[str, Any]] = []

    methods = _resolve_graphrag_methods()
    trace["official_graphrag"]["methods"] = methods

    for method in methods:
        text, q_trace = _run_graphrag_cli_query(query=query, method=method)
        official_traces.append(q_trace)
        if text:
            official_outputs.append(
                {
                    "method": method,
                    "text": text,
                }
            )

    trace["official_graphrag"]["results"] = official_traces
    official_ok = bool(official_outputs)

    proxy_hits: list[dict[str, Any]] = []
    if not official_ok:
        if bool(getattr(settings, "graphrag_require_official", False)):
            trace["proxy_fallback"] = {
                "skipped": True,
                "reason": "GRAPHRAG_REQUIRE_OFFICIAL=1",
            }
        else:
            proxy_hits, proxy_trace = _proxy_graph_retrieve(
                query=query,
                corpus=corpus,
                visual_summary=visual_summary,
            )
            trace["proxy_fallback"] = proxy_trace

    context = _build_context(
        question=question,
        visual_summary=visual_summary,
        official_outputs=official_outputs,
        proxy_hits=proxy_hits,
    )

    evidence = _evidence_from_outputs(
        visual_summary=visual_summary,
        official_outputs=official_outputs,
        proxy_hits=proxy_hits,
    )

    answer = ""
    error = None

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
        if callable(map_answer_fn):
            answer = str(map_answer_fn(answer)).strip()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    hit_count = len(official_outputs) + len(proxy_hits)
    confidence = 0.35
    if visual_summary:
        confidence += 0.12
    if official_outputs:
        confidence += min(0.35, 0.13 * len(official_outputs))
    if proxy_hits:
        confidence += min(0.22, 0.035 * len(proxy_hits))
    if not answer:
        confidence = min(confidence, 0.35)

    decision_status = "finalized" if answer else "fallback"
    decision_reason = "graphrag_mllm"
    if official_outputs:
        decision_reason = "official_graphrag_mllm"
    elif proxy_hits:
        decision_reason = "proxy_graph_fallback_mllm"
    elif error:
        decision_reason = "graphrag_mllm_answer_error"

    return {
        "mode": "graphrag_mllm",
        "answer": answer,
        "context": context,
        "evidence": evidence,
        "tool_logs": [],
        "retrieval_trace": trace,
        "kg_trace": {
            "provider": "microsoft_graphrag_cli" if official_outputs else "lexical_graph_fallback",
            "official_ok": bool(official_outputs),
            "hit_count": hit_count,
            "methods": methods,
        },
        "raw_hits": official_outputs if official_outputs else proxy_hits,
        "error": error,
        "final_confidence": _safe_float(confidence, 0.0),
        "decision_status": decision_status,
        "decision_reason": decision_reason,
    }