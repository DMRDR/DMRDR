from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import math
import mimetypes
import os
import random
import re
import shutil
import time


from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

from config import settings
from openai import OpenAI

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable: Any, **_: Any) -> Any:  # type: ignore[misc]
        return iterable


from difficulty import _task_family, _qnorm_100, difficulty_bucket, estimate_question_difficulty
from lexical_graph import LexicalGraphRetriever, build_graph_from_corpus

# DeepResearch / agentic imports
from agents import (
    CriticAgent,
    Evidence,
    PlannerAgent,
    ResearchState,
    ResearcherAgent,
    SubQuestion,
    SynthesizerAgent,
    VerifierAgent,
)
from tools import (
    BYOKGRAGProvider,
    LexicalGraphTool,
    SearchTool,
    ToolObservation,
    ToolRouter,
    VisionTool,
    WebpageTool,
)
from workflow import DeepResearchWorkflow


@dataclass
class AblationSetting:
    use_kg_rag: bool
    use_difficulty: bool
    use_sft: bool
    profile: str


@dataclass
class ExecutorOutput:
    mode: str
    answer: str
    context: str = ""
    evidence: list[dict[str, Any]] = field(default_factory=list)
    tool_logs: list[dict[str, Any]] = field(default_factory=list)
    retrieval_trace: dict[str, Any] = field(default_factory=dict)
    kg_trace: dict[str, Any] = field(default_factory=dict)
    raw_hits: list[Any] = field(default_factory=list)
    error: str | None = None
    final_confidence: float = 0.0
    decision_status: str = "finalized"  # finalized / abstained / fallback
    decision_reason: str = ""

def _get_vllm_local_media_root(args: argparse.Namespace | None = None) -> Path | None:
    if args is not None:
        value = getattr(args, "vllm_local_media_root", None)
        if value:
            return Path(value).expanduser().resolve()

    env_value = os.getenv("VLLM_ALLOWED_LOCAL_MEDIA_ROOT", "").strip()
    if env_value:
        return Path(env_value).expanduser().resolve()

    return None


def _dataset_media_bucket(sample: dict[str, Any]) -> str:
    dataset_type = str(sample.get("dataset_type", "generic")).strip().lower()
    if dataset_type in {"aokvqa", "m3cot", "mmmu_pro", "scienceqa", "cmmqa"}:
        return dataset_type
    return "generic"


def _to_file_url(path: Path) -> str:
    return f"file://{path.resolve().as_posix()}"


def _require_vllm_compatible_image_ref(image_ref: str | None) -> str | None:
    ref = str(image_ref or "").strip()
    if not ref:
        return None
    if ref.startswith("data:image/"):
        raise ValueError(
            "Unexpected inline data URL image_ref. "
            "Please route all image resolution through resolve_image_reference()."
        )
    return ref


def _persist_image_bytes_to_vllm_media_root(
    img_bytes: bytes,
    media_root: Path,
    bucket: str,
    ext: str = ".jpg",
) -> str:
    target_dir = media_root / bucket / "image_bytes"
    target_dir.mkdir(parents=True, exist_ok=True)

    ext = str(ext or ".jpg").strip().lower()
    if not ext.startswith("."):
        ext = "." + ext
    if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
        ext = ".jpg"

    digest = hashlib.sha1(img_bytes).hexdigest()
    out_path = target_dir / f"{digest}{ext}"

    if not out_path.exists():
        out_path.write_bytes(img_bytes)

    return _to_file_url(out_path)


def _stage_local_image_to_vllm_media_root(
    src_path: Path,
    media_root: Path,
    bucket: str,
) -> str:
    target_dir = media_root / bucket / "local_files"
    target_dir.mkdir(parents=True, exist_ok=True)

    content = src_path.read_bytes()
    digest = hashlib.sha1(content).hexdigest()

    ext = src_path.suffix.lower().strip() or ".jpg"
    if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
        ext = ".jpg"

    out_path = target_dir / f"{digest}{ext}"

    if not out_path.exists():
        shutil.copy2(src_path, out_path)

    return _to_file_url(out_path)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
    except Exception:
        return default
    if math.isnan(x):
        return default
    return x


def _mean_or_zero(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _extract_structured_vision_summary(obj: dict[str, Any]) -> str:
    if not isinstance(obj, dict):
        return ""

    parts: list[str] = []

    visible_facts = obj.get("visible_facts", {})
    if isinstance(visible_facts, dict) and visible_facts:
        fact_lines = [f"{k}: {v}" for k, v in visible_facts.items()]
        parts.append("可见事实:\n" + "\n".join(fact_lines[:12]))

    ocr_facts = obj.get("ocr_facts", [])
    if isinstance(ocr_facts, list) and ocr_facts:
        ocr_lines: list[str] = []
        for item in ocr_facts[:10]:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            region = str(item.get("region", "unknown")).strip()
            conf = item.get("confidence", "")
            if text:
                ocr_lines.append(f"[{region}] {text} (confidence={conf})")
        if ocr_lines:
            parts.append("OCR事实:\n" + "\n".join(ocr_lines))

    ambiguities = obj.get("ambiguities", [])
    if isinstance(ambiguities, list) and ambiguities:
        amb_lines = [str(x).strip() for x in ambiguities[:8] if str(x).strip()]
        if amb_lines:
            parts.append("歧义:\n" + "\n".join(amb_lines))

    hypotheses = obj.get("hypotheses", [])
    if isinstance(hypotheses, list) and hypotheses:
        hyp_lines: list[str] = []
        for item in hypotheses[:4]:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", "")).strip()
            reason = str(item.get("reason", "")).strip()
            conf = item.get("confidence", "")
            if label:
                hyp_lines.append(f"{label} | reason={reason} | confidence={conf}")
        if hyp_lines:
            parts.append("候选解释:\n" + "\n".join(hyp_lines))

    return "\n\n".join(parts)[:2000]


def _fallback_decision_confidence_from_evidence(evidence: list[dict[str, Any]]) -> float:
    """
    仅在 agents.decide()/critic.final_confidence 缺失时使用的回退置信度。
    """
    if not evidence:
        return 0.0

    confs = [_safe_float(e.get("confidence", 0.0), 0.0) for e in evidence]
    exclusivities = [_safe_float(e.get("exclusivity", 0.0), 0.0) for e in evidence]
    ambiguities = [_safe_float(e.get("ambiguity", 0.0), 0.0) for e in evidence]
    premise_penalties = [1.0 if e.get("premise_dependencies") else 0.0 for e in evidence]

    score = (
        0.50 * _mean_or_zero(confs)
        + 0.35 * _mean_or_zero(exclusivities)
        - 0.25 * _mean_or_zero(ambiguities)
        - 0.15 * _mean_or_zero(premise_penalties)
    )
    return max(0.0, min(1.0, score))


def _has_contradiction_marker(text: str) -> bool:
    low = str(text or "").lower()
    markers = [
        " not ",
        "not actually",
        "rather than",
        "instead of",
        "used as props",
        "staged",
        "不是",
        "并非",
        "不是真正",
        "不是真的",
        "并不是在",
        "更像是",
        "只是道具",
    ]
    return any(m in low for m in markers)


def _structured_visual_to_seed_evidence(
    structured: dict[str, Any],
    source: str = "local_image",
) -> list[Evidence]:
    out: list[Evidence] = []
    if not isinstance(structured, dict):
        return out

    visible_facts = structured.get("visible_facts", {})
    if isinstance(visible_facts, dict) and visible_facts:
        fact_excerpt = "\n".join([f"{k}: {v}" for k, v in visible_facts.items()][:12])
        out.append(
            Evidence(
                source=source,
                claim="图像可见事实",
                excerpt=fact_excerpt,
                confidence=0.78,
                evidence_type="image_fact",
                premise_dependencies=[],
                ambiguity=0.05,
                exclusivity=0.35,
            )
        )

    ocr_facts = structured.get("ocr_facts", [])
    if isinstance(ocr_facts, list) and ocr_facts:
        ocr_lines: list[str] = []
        for item in ocr_facts[:10]:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            region = str(item.get("region", "unknown")).strip()
            conf = _safe_float(item.get("confidence", 0.75), 0.75)
            if text:
                ocr_lines.append(f"[{region}] {text} (confidence={conf})")
        if ocr_lines:
            out.append(
                Evidence(
                    source=f"{source} # ocr",
                    claim="图像OCR事实",
                    excerpt="\n".join(ocr_lines),
                    confidence=0.82,
                    evidence_type="ocr",
                    premise_dependencies=[],
                    ambiguity=0.08,
                    exclusivity=0.45,
                )
            )

    hypotheses = structured.get("hypotheses", [])
    if isinstance(hypotheses, list) and hypotheses:
        for idx, item in enumerate(hypotheses[:4], start=1):
            if not isinstance(item, dict):
                continue

            label = str(item.get("label", "")).strip()
            reason = str(item.get("reason", "")).strip()
            conf = _safe_float(item.get("confidence", 0.55), 0.55)
            merged = f"{label} {reason}".strip()
            if not merged:
                continue

            is_contradiction = _has_contradiction_marker(merged)
            out.append(
                Evidence(
                    source=f"{source} # hypothesis",
                    claim=label or f"图像候选解释{idx}",
                    excerpt=reason or merged[:500],
                    confidence=max(0.45, min(conf, 0.98)),
                    supports=[],
                    contradicts=["surface_interpretation"] if is_contradiction else [],
                    evidence_type="hypothesis",
                    premise_dependencies=[] if is_contradiction else ["需要将图像候选解释与其他证据交叉验证"],
                    ambiguity=0.25 if is_contradiction else 0.40,
                    exclusivity=0.52 if is_contradiction else 0.18,
                )
            )

    ambiguities = structured.get("ambiguities", [])
    if isinstance(ambiguities, list) and ambiguities:
        amb_excerpt = "\n".join([str(x).strip() for x in ambiguities[:8] if str(x).strip()])
        if amb_excerpt:
            out.append(
                Evidence(
                    source=f"{source} # ambiguity",
                    claim="图像仍存在歧义",
                    excerpt=amb_excerpt,
                    confidence=0.70,
                    evidence_type="ambiguity",
                    premise_dependencies=[],
                    ambiguity=0.95,
                    exclusivity=0.0,
                )
            )

    return out


def _structured_visual_suggested_followups(structured: dict[str, Any]) -> list[str]:
    if not isinstance(structured, dict):
        return []
    followups = structured.get("suggested_followups", [])
    if not isinstance(followups, list):
        return []
    out = [str(x).strip() for x in followups if str(x).strip()]
    return out[:3]


def _structured_visual_to_seed_docs(
    structured: dict[str, Any],
    source: str = "local_image_file",
) -> list[dict[str, str]]:
    docs: list[dict[str, str]] = []
    if not isinstance(structured, dict):
        return docs

    visible_facts = structured.get("visible_facts", {})
    if isinstance(visible_facts, dict) and visible_facts:
        fact_lines = [f"{k}: {v}" for k, v in visible_facts.items()]
        docs.append(
            {
                "source": source,
                "text": "图像可见事实:\n" + "\n".join(fact_lines[:12]),
            }
        )

    ocr_facts = structured.get("ocr_facts", [])
    if isinstance(ocr_facts, list) and ocr_facts:
        ocr_lines: list[str] = []
        for item in ocr_facts[:10]:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            region = str(item.get("region", "unknown")).strip()
            conf = item.get("confidence", "")
            if text:
                ocr_lines.append(f"[{region}] {text} (confidence={conf})")
        if ocr_lines:
            docs.append(
                {
                    "source": f"{source} # ocr_facts",
                    "text": "图像OCR事实:\n" + "\n".join(ocr_lines),
                }
            )

    return docs


def _detect_unresolved_conflict(evidence: list[dict[str, Any]]) -> bool:
    if not evidence:
        return False

    support_scores: list[float] = []
    contradiction_scores: list[float] = []

    for e in evidence:
        source = str(e.get("source", "")).strip().lower()
        etype = str(e.get("evidence_type", "")).strip().lower()
        merged = f"{e.get('claim', '')} {e.get('excerpt', '')}".strip()
        conf = _safe_float(e.get("confidence", 0.0), 0.0)
        ambiguity = _safe_float(e.get("ambiguity", 0.0), 0.0)
        exclusivity = _safe_float(e.get("exclusivity", 0.0), 0.0)

        supports = e.get("supports", [])
        contradicts = e.get("contradicts", [])
        if not isinstance(supports, list):
            supports = []
        if not isinstance(contradicts, list):
            contradicts = []

        is_visual = (
            etype in {"image_fact", "ocr", "hypothesis"}
            or "local_image" in source
            or "vision_recheck" in source
            or "image" in source
        )

        if conf >= 0.55 and ambiguity <= 0.65:
            if supports:
                support_scores.append(conf + 0.15 * exclusivity)
            elif etype in {"image_fact", "ocr", "graph", "kb_chunk", "webpage_fact", "text"} and not _has_contradiction_marker(merged):
                support_scores.append(conf + 0.10 * exclusivity)

        if is_visual and conf >= 0.70:
            if contradicts or _has_contradiction_marker(merged):
                contradiction_scores.append(conf + 0.10 * (1.0 - ambiguity))

    return bool(support_scores and contradiction_scores)


def _should_finalize_agent_answer(
    *,
    critique: Any,
    serialized_evidence: list[dict[str, Any]],
    decision_confidence: float,
    normalized_answer: str,
    unresolved_conflict: bool,
) -> tuple[bool, str]:
    if not normalized_answer or normalized_answer.upper() == "ABSTAIN":
        return False, "empty_or_abstain_answer"

    if critique is None:
        return False, "missing_critique"

    if bool(getattr(critique, "must_abort_finalization", False)):
        return False, "critic_force_abort"

    if not bool(getattr(critique, "sufficient", False)):
        return False, "insufficient_evidence"

    if unresolved_conflict:
        return False, "unresolved_conflict"

    threshold = float(getattr(settings, "final_answer_confidence_threshold", 0.60) or 0.60)
    if float(decision_confidence) < threshold:
        return False, "low_confidence"

    strong_support = 0
    for e in serialized_evidence:
        conf = _safe_float(e.get("confidence", 0.0), 0.0)
        ambiguity = _safe_float(e.get("ambiguity", 0.0), 0.0)
        etype = str(e.get("evidence_type", "")).strip().lower()
        supports = e.get("supports", [])
        if not isinstance(supports, list):
            supports = []

        if conf >= 0.60 and ambiguity <= 0.55:
            if supports or etype in {"image_fact", "ocr", "graph", "kb_chunk", "webpage_fact"}:
                strong_support += 1

    if strong_support <= 0:
        return False, "no_relevant_support"

    return True, "finalized"

def _resolve_ablation(profile: str) -> AblationSetting:
    table = {
        "none": AblationSetting(False, False, False, "none"),
        "kg_only": AblationSetting(True, False, False, "kg_only"),
        "difficulty_only": AblationSetting(False, True, False, "difficulty_only"),
        "kg_difficulty": AblationSetting(True, True, False, "kg_difficulty"),
        "kg_sft": AblationSetting(True, False, True, "kg_sft"),
        "difficulty_sft": AblationSetting(False, True, True, "difficulty_sft"),
        "all_on": AblationSetting(True, True, True, "all_on"),
    }
    return table.get(profile, table["all_on"])


def _resolve_effective_web_search_tool_enabled(
    args: argparse.Namespace,
    ablation: AblationSetting,
) -> bool:
    explicit = getattr(args, "enable_web_search_tool", None)
    if explicit is not None:
        return bool(explicit)

    if not bool(getattr(settings, "enable_web_search_tool", True)):
        return False

    disabled_profiles = {
        str(x).strip().lower()
        for x in getattr(settings, "no_web_search_ablation_profiles", []) or []
        if str(x).strip()
    }

    profile = str(getattr(ablation, "profile", "") or "").strip().lower()
    if profile in disabled_profiles:
        return False

    return True


def _jsonable(v: Any) -> Any:
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, (bytes, bytearray)):
        b = bytes(v)
        return {
            "__type": "bytes",
            "size": len(b),
            "sha1": hashlib.sha1(b).hexdigest(),
        }
    if isinstance(v, dict):
        return {str(k): _jsonable(val) for k, val in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_jsonable(x) for x in v]
    if hasattr(v, "tolist"):
        try:
            return _jsonable(v.tolist())
        except Exception:
            pass
    if hasattr(v, "item"):
        try:
            return _jsonable(v.item())
        except Exception:
            pass
    return str(v)


def _record_key(sample: dict[str, Any]) -> str:
    original = sample.get("__original_record")
    if isinstance(original, dict):
        for k in ["id", "question_id", "image_id"]:
            v = original.get(k)
            if v is not None and str(v).strip():
                return f"id::{k}::{v}"
    text = f"{sample.get('question', '')}||{sample.get('answer', '')}"
    return "sha1::" + hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def _load_resume_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.suffix.lower() == ".jsonl":
        out: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                out.append(obj)
        return out
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    return []


def _tertile_thresholds(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.34, 0.67
    arr = sorted(values)

    def _q(p: float) -> float:
        if len(arr) == 1:
            return arr[0]
        idx = p * (len(arr) - 1)
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            return arr[lo]
        frac = idx - lo
        return arr[lo] * (1.0 - frac) + arr[hi] * frac

    return _q(1.0 / 3.0), _q(2.0 / 3.0)


def normalize(s: str) -> str:
    return re.sub(r"\s+", "", s.lower())

_MC_DATASETS = {"aokvqa", "m3cot", "mmmu_pro", "cmmqa", "scienceqa"}

_BLOCKED_RETRIEVAL_SOURCES = {
    "scienceqa": {"solution"},
    "mmmu_pro": {"solution"},
}

_SYNTHETIC_SOURCES = {"choice_hint", "question_type", "common_sense"}
_RETRIEVAL_ANNOTATION_SOURCES = {
    "rationale",
    "explain",
    "explanation",
    "reasoning_chains",
}


def _normalize_cmp_text(s: str) -> str:
    s = "" if s is None else str(s)
    return re.sub(r"[\s\.\,\:\;\!\?\-\_\(\)\[\]\{\}\"\'`]+", "", s.lower())

def _extract_choice_label_from_text(text: str, max_choices: int) -> int | None:
    if not text:
        return None
    s = str(text).strip().upper()

    # 完全是单字母
    if len(s) == 1 and "A" <= s <= "Z":
        idx = ord(s) - ord("A")
        return idx if 0 <= idx < max_choices else None

    # (A) / [A]
    m = re.search(r"[\(\[]\s*([A-Z])\s*[\)\]]", s)
    if m:
        idx = ord(m.group(1)) - ord("A")
        return idx if 0 <= idx < max_choices else None

    # 独立字母 token
    m = re.search(r"\b([A-Z])\b", s)
    if m:
        idx = ord(m.group(1)) - ord("A")
        return idx if 0 <= idx < max_choices else None

    return None

def _match_choice_text(pred: str, choices: list[str]) -> int | None:
    if not pred or not choices:
        return None

    pn = _normalize_cmp_text(pred)
    if not pn:
        return None

    # 先做完全匹配
    for i, ch in enumerate(choices):
        cn = _normalize_cmp_text(ch)
        if cn and pn == cn:
            return i

    # 再做选项文本包含于生成句匹配
    matched: list[tuple[int, int]] = []
    for i, ch in enumerate(choices):
        cn = _normalize_cmp_text(ch)
        if cn and cn in pn:
            matched.append((i, len(cn)))

    if matched:
        # 选最长的那个，避免短字符串误命中
        matched.sort(key=lambda x: x[1], reverse=True)
        return matched[0][0]

    return None

def canonicalize_eval_answer(answer: str, sample: dict[str, Any]) -> str:
    raw = "" if answer is None else str(answer).strip()
    dataset_type = str(sample.get("dataset_type", "")).lower()
    choices = _to_choice_list(sample.get("choices"))

    # 非选择题完全不变
    if dataset_type not in _MC_DATASETS or not choices:
        return raw

    # 先看能不能从字母还原
    idx = _extract_choice_label_from_text(raw, len(choices))
    if idx is not None:
        return str(choices[idx]).strip()

    # 再看能不能从文本匹配到选项
    idx = _match_choice_text(raw, choices)
    if idx is not None:
        return str(choices[idx]).strip()

    # 匹配不到就原样返回
    return raw

def exact_match_for_eval(pred: str, gold: str, sample: dict[str, Any]) -> int:
    pred_c = canonicalize_eval_answer(pred, sample)
    gold_c = canonicalize_eval_answer(gold, sample)
    return int(_normalize_cmp_text(pred_c) == _normalize_cmp_text(gold_c))

def f1_for_eval(pred: str, gold: str, sample: dict[str, Any]) -> float:
    pred_c = canonicalize_eval_answer(pred, sample)
    gold_c = canonicalize_eval_answer(gold, sample)
    return token_f1(pred_c, gold_c)


def token_f1(pred: str, gold: str) -> float:
    p = list(normalize(pred))
    g = list(normalize(gold))
    if not p or not g:
        return 0.0
    common = 0
    g_used = [False] * len(g)
    for ch in p:
        for i, gg in enumerate(g):
            if not g_used[i] and ch == gg:
                g_used[i] = True
                common += 1
                break
    if common == 0:
        return 0.0
    precision = common / len(p)
    recall = common / len(g)
    return 2 * precision * recall / (precision + recall)


def _contains_answer(text: str, answer: str) -> bool:
    text = "" if text is None else str(text)
    answer = "" if answer is None else str(answer).strip()
    if not answer:
        return False

    t_norm = normalize(text)
    a_norm = normalize(answer)
    if not a_norm:
        return False

    # 中文或较长答案仍允许归一化后的包含匹配
    if re.search(r"[\u4e00-\u9fff]", answer) or len(a_norm) >= 6:
        return a_norm in t_norm

    # 选择题/短英文答案尽量做边界约束，避免 over / red / cat 之类误命中
    text_low = text.lower()
    ans_low = answer.lower().strip()

    # 优先按独立词/短语匹配
    pattern = r"(?<![A-Za-z0-9_])" + re.escape(ans_low) + r"(?![A-Za-z0-9_])"
    if re.search(pattern, text_low):
        return True

    # 如果答案本身是选项文本且较短，再退回一次严格归一化相等/片段匹配
    return a_norm == t_norm or f" {ans_low} " in f" {text_low} "


def compute_evidence_metrics(graph_hits: list[Any], docs: list[dict[str, Any]], answer: str) -> tuple[int, float, float]:
    if not answer:
        return 0, 0.0, 0.0

    relevant_total = 0
    for d in docs:
        txt = str(d.get("text", ""))
        if txt and _contains_answer(txt, answer):
            relevant_total += 1

    retrieved_relevant = 0
    for h in graph_hits:
        txt = str(getattr(h, "text", ""))
        if txt and _contains_answer(txt, answer):
            retrieved_relevant += 1

    retrieved_total = len(graph_hits)
    hits = int(retrieved_relevant > 0)
    recall = (retrieved_relevant / relevant_total) if relevant_total > 0 else 0.0
    precision = (retrieved_relevant / retrieved_total) if retrieved_total > 0 else 0.0
    return hits, recall, precision


def compute_serialized_evidence_metrics(
    evidence: list[dict[str, Any]],
    answer: str,
    supporting_docs: list[dict[str, Any]] | None = None,
) -> tuple[int, float, float]:
    if not answer or not evidence:
        return 0, 0.0, 0.0

    support_hits = 0
    contradiction_hits = 0

    for e in evidence:
        claim = str(e.get("claim", ""))
        excerpt = str(e.get("excerpt", ""))
        merged = f"{claim} {excerpt}"

        supports = e.get("supports", [])
        contradicts = e.get("contradicts", [])

        if not isinstance(supports, list):
            supports = []
        if not isinstance(contradicts, list):
            contradicts = []

        support_match = _contains_answer(merged, answer) or any(_contains_answer(str(x), answer) for x in supports)
        contradiction_match = any(_contains_answer(str(x), answer) for x in contradicts)

        if support_match:
            support_hits += 1
        if contradiction_match:
            contradiction_hits += 1

    retrieved_total = len(evidence)
    hits = int(support_hits > 0)
    precision = (support_hits / retrieved_total) if retrieved_total > 0 else 0.0

    denom = support_hits + contradiction_hits
    if denom > 0:
        recall = support_hits / denom
    else:
        recall = 1.0 if hits else 0.0

    return hits, recall, precision


def compute_evidence_metrics_v2(
    gold: str,
    pred: str,
    evidence: list[dict[str, Any]],
    sample: dict[str, Any],
) -> dict[str, float]:
    if not evidence:
        return {
            "evidence_count": 0.0,
            "support_hit": 0.0,
            "avg_confidence": 0.0,
            "answer_supported": 0.0,
            "exclusive_support": 0.0,
            "ambiguity_penalty": 0.0,
            "premise_dependency": 0.0,
            "hard_negative_supported": 0.0,
        }

    gold_c = canonicalize_eval_answer(gold, sample)
    pred_c = canonicalize_eval_answer(pred, sample)

    confs: list[float] = []
    ambiguities: list[float] = []
    exclusivities: list[float] = []
    premise_flags: list[float] = []

    support_hit = 0
    answer_supported = 0
    exclusive_support = 0.0
    hard_negative_supported = 0.0

    all_choices = _to_choice_list(sample.get("choices"))
    normalized_choices = [canonicalize_eval_answer(x, sample) for x in all_choices]

    for e in evidence:
        excerpt = str(e.get("excerpt", ""))
        claim = str(e.get("claim", ""))
        merged = f"{excerpt} {claim}".strip()

        conf = _safe_float(e.get("confidence", 0.0), 0.0)
        ambiguity = _safe_float(e.get("ambiguity", 0.0), 0.0)
        exclusivity = _safe_float(e.get("exclusivity", 0.0), 0.0)
        premise_deps = e.get("premise_dependencies", [])
        supports = e.get("supports", [])
        contradicts = e.get("contradicts", [])

        if not isinstance(premise_deps, list):
            premise_deps = []
        if not isinstance(supports, list):
            supports = []
        if not isinstance(contradicts, list):
            contradicts = []

        confs.append(conf)
        ambiguities.append(ambiguity)
        exclusivities.append(exclusivity)
        premise_flags.append(1.0 if premise_deps else 0.0)

        if _contains_answer(merged, gold_c):
            support_hit = 1

        pred_hit = False
        if pred_c:
            if _contains_answer(merged, pred_c):
                pred_hit = True
            elif any(_normalize_cmp_text(str(x)) == _normalize_cmp_text(pred_c) for x in supports):
                pred_hit = True

        if pred_hit:
            answer_supported = 1
            exclusive_support = max(exclusive_support, exclusivity)

        if pred_c and pred_hit:
            competing = [x for x in normalized_choices if x and _normalize_cmp_text(x) != _normalize_cmp_text(pred_c)]
            if any(_contains_answer(merged, c) for c in competing):
                hard_negative_supported = 1.0
            elif any(_normalize_cmp_text(str(x)) in {_normalize_cmp_text(c) for c in competing} for x in supports):
                hard_negative_supported = 1.0

        if pred_c and any(_normalize_cmp_text(str(x)) == _normalize_cmp_text(pred_c) for x in contradicts):
            hard_negative_supported = 1.0

    return {
        "evidence_count": float(len(evidence)),
        "support_hit": float(support_hit),
        "avg_confidence": _mean_or_zero(confs),
        "answer_supported": float(answer_supported),
        "exclusive_support": float(exclusive_support),
        "ambiguity_penalty": _mean_or_zero(ambiguities),
        "premise_dependency": _mean_or_zero(premise_flags),
        "hard_negative_supported": float(hard_negative_supported),
    }


# 全局变量存储动态阈值（由数据集难度分布计算得出）
_route_thresholds = {"q33": 0.34, "q67": 0.67}


def set_route_thresholds(q33: float, q67: float) -> None:
    global _route_thresholds
    _route_thresholds = {"q33": q33, "q67": q67}


def _difficulty_route_tag(overall: float, use_dynamic: bool = True) -> str:
    global _route_thresholds
    q33 = _route_thresholds.get("q33", 0.34)
    q67 = _route_thresholds.get("q67", 0.67)

    if overall > q67:
        return "full_rag"
    if overall > q33:
        return "light_rag"
    return "direct_inference"


def _sft_role(bucket: str, verification: str) -> str:
    if bucket == "hard" and verification == "Unverified":
        return "hard_unverified_primary"
    if bucket == "hard" and verification == "Verified":
        return "hard_verified_aux"
    return "not_selected"


def _build_route_aware_graph_context(
    route: str,
    question: str,
    corpus: list[tuple[str, str]],
) -> tuple[list[Any], str, dict[str, Any]]:
    if route == "direct_inference":
        return [], "", {
            "rounds": 0,
            "top_k": 0,
            "expanded_query": None,
            "used_raw_fallback": False,
        }

    if not corpus:
        return [], "", {
            "rounds": 1 if route == "light_rag" else 2,
            "top_k": 0,
            "expanded_query": None,
            "used_raw_fallback": False,
        }

    index = build_graph_from_corpus(corpus)
    retriever = LexicalGraphRetriever(index)

    if route == "light_rag":
        graph_hits = retriever.retrieve(question, top_k=3)

        hit_lines = [
            f"[{h.source}] {str(h.text)[:600]}"
            for h in graph_hits
            if str(getattr(h, "text", "")).strip()
        ]
        raw_lines = [
            f"[{source}] {str(text)[:600]}"
            for source, text in corpus[:4]
            if str(text).strip()
        ]

        used_raw_fallback = False
        if hit_lines:
            # 命中时也保留少量原始文档，避免 proxy graph 只看到过窄片段
            graph_context = "\n\n".join(hit_lines + raw_lines[:2])
        else:
            # lexical miss 时直接退回原始 corpus，避免出现 proxy_graph_empty_context
            graph_context = "\n\n".join(raw_lines)
            used_raw_fallback = True

        return graph_hits, graph_context[:5000], {
            "rounds": 1,
            "top_k": 3,
            "expanded_query": None,
            "used_raw_fallback": used_raw_fallback,
        }

    first_hits = retriever.retrieve(question, top_k=5)
    expansion_terms: list[str] = []
    for hit in first_hits[:2]:
        snippet = str(getattr(hit, "text", ""))[:120]
        if snippet:
            expansion_terms.append(snippet)

    expanded_query = question if not expansion_terms else f"{question}\n补充证据: {' '.join(expansion_terms)}"
    second_hits = retriever.retrieve(expanded_query, top_k=5)

    merged: list[Any] = []
    seen = set()
    for hit in [*first_hits, *second_hits]:
        key = (getattr(hit, "source", ""), getattr(hit, "text", ""))
        if key in seen:
            continue
        seen.add(key)
        merged.append(hit)

    graph_context = "\n\n".join([f"[{h.source}] {h.text[:600]}" for h in merged[:8]])
    return merged[:8], graph_context, {
        "rounds": 2,
        "top_k": 5,
        "expanded_query": expanded_query if expansion_terms else None,
        "used_raw_fallback": False,
    }


def _is_missing(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    try:
        import pandas as pd
        if isinstance(v, float) and pd.isna(v):
            return True
    except ImportError:
        pass
    try:
        if v != v:
            return True
    except Exception:
        pass
    return False


def _extract_answer(record: dict[str, Any], args: argparse.Namespace) -> str:
    v = record.get(args.answer_field)
    if not _is_missing(v):
        return str(v)

    direct = record.get("direct_answers")
    if direct is not None:
        if hasattr(direct, "tolist"):
            direct = direct.tolist()
        if isinstance(direct, list) and len(direct) > 0:
            valid_answers = [str(x) for x in direct if not _is_missing(x)]
            if valid_answers:
                most_common = Counter(valid_answers).most_common(1)[0][0]
                return str(most_common)

    idx = record.get("correct_choice_idx")
    choices = record.get("choices")
    if not _is_missing(idx) and choices is not None:
        try:
            i_val = int(idx)
            if hasattr(choices, "to_pylist"):
                choices = choices.to_pylist()
            elif hasattr(choices, "tolist"):
                choices = choices.tolist()
            if isinstance(choices, (list, tuple)) and 0 <= i_val < len(choices):
                return str(choices[i_val])
        except Exception:
            pass

    raise ValueError("Unable to infer answer; set --answer-field or provide direct_answers / correct_choice_idx+choices.")


def _to_choice_list(v: Any) -> list[str]:
    if v is None:
        return []
    if hasattr(v, "tolist"):
        try:
            v = v.tolist()
        except Exception:
            pass
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return []
        try:
            parsed = ast.literal_eval(s)
        except (SyntaxError, ValueError):
            return []
        if isinstance(parsed, (list, tuple)):
            return [str(x) for x in parsed]
    return []



def _image_ref_to_data_url_for_retrieval_seed(llm: Any, image_ref: str | None) -> str:
    if llm is None:
        return ""
    ref = str(image_ref or "").strip()
    if not ref:
        return ""
    if ref.startswith("data:image/"):
        return ref
    if hasattr(llm, "image_ref_to_data_url"):
        try:
            return str(llm.image_ref_to_data_url(ref) or "")
        except Exception:
            return ""
    return ""


def _build_gpt_retrieval_seed_prompt(question: str, choices: list[str], dataset_type: str) -> str:
    choices_block = "\n".join([f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(choices[:10])]) if choices else "(no choices)"
    return (
        "假设你是一个专业的检索语料增强助手，你绝不允许直接答题。\n"
        "请基于题目、选项和图片，生成后续检索阶段最有用的辅助语料。\n"
        "严格只输出一个 JSON 对象，不要输出任何额外解释。\n"
        "JSON schema:\n"
        "{\n"
        '  "retrieval_queries": ["..."],\n'
        '  "retrieval_hints": ["..."],\n'
        '  "visual_facts": ["..."],\n'
        '  "option_discriminators": ["..."],\n'
        '  "avoid_topics": ["..."]\n'
        "}\n"
        "要求：\n"
        "1) 不要直接给最终答案；\n"
        "2) retrieval_queries 应尽量短，适合搜索和知识库检索；\n"
        "3) visual_facts 只写可以从图像直接观察或 OCR 提取的事实；\n"
        "4) option_discriminators 只写真正能区分候选项的判别线索；\n"
        "5) avoid_topics 写出容易把检索带偏的主题。\n\n"
        f"dataset_type: {dataset_type}\n"
        f"question:\n{question}\n\n"
        f"choices:\n{choices_block}"
    )
    

def _gpt_retry_attempts() -> int:
    try:
        n = int(getattr(settings, "api_retries", 3) or 3)
    except Exception:
        n = 3
    return max(1, min(n, 3))


def _gpt_retry_sleep_s(attempt_idx: int) -> float:
    try:
        base = float(getattr(settings, "api_retry_backoff_s", 2) or 2)
    except Exception:
        base = 2.0
    try:
        cap = float(getattr(settings, "api_retry_max_wait_s", 120) or 120)
    except Exception:
        cap = 120.0
    return max(0.5, min(base * (2 ** attempt_idx), cap, 120.0))


from tools import parse_loose_json_object
def _call_xiaoai_gpt_retrieval_seed(
    *,
    question: str,
    choices: list[str],
    dataset_type: str,
    image_data_url: str = "",
    llm: Any = None,
) -> dict[str, Any]:
    if not bool(getattr(settings, "enable_gpt_retrieval_seed", False)):
        return {"ok": False, "error": "disabled"}

    api_base = str(getattr(settings, "gpt_retrieval_seed_api_base", "") or "").strip()
    api_key = str(getattr(settings, "gpt_retrieval_seed_api_key", "") or "").strip()
    model = str(getattr(settings, "gpt_retrieval_seed_model", "gpt-5.1") or "gpt-5.1").strip()

    if not api_base or not api_key:
        return {"ok": False, "error": "missing_api_base_or_key"}

    client = OpenAI(
        base_url=api_base,
        api_key=api_key,
        timeout=float(getattr(settings, "gpt_retrieval_seed_timeout_s", 90) or 90),
    )

    prompt = _build_gpt_retrieval_seed_prompt(
        question=question,
        choices=choices,
        dataset_type=dataset_type,
    )

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    if image_data_url and bool(getattr(settings, "gpt_retrieval_seed_include_image", True)):
        content.append({"type": "image_url", "image_url": {"url": image_data_url}})

    attempts = _gpt_retry_attempts()
    raw = ""
    last_error: Exception | None = None

    for i in range(attempts):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.1,
                max_tokens=int(getattr(settings, "gpt_retrieval_seed_max_tokens", 2048) or 1200),
                messages=[
                    {"role": "user", "content": content},
                ],
            )
            raw = str(resp.choices[0].message.content or "").strip()
            last_error = None
            break
        except Exception as exc:
            last_error = exc
            if i < attempts - 1:
                time.sleep(_gpt_retry_sleep_s(i))
                continue
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    if last_error is not None and not raw:
        return {"ok": False, "error": f"{type(last_error).__name__}: {last_error}"}

    try:
        parsed = parse_loose_json_object(raw, fallback={})
    except Exception:
        parsed = {}

    if not isinstance(parsed, dict):
        return {"ok": False, "error": "parse_failed", "raw": raw[:3000]}

    parsed["_raw_model_output"] = raw[:3000]
    return {"ok": True, "data": parsed}


def _parse_gpt_retrieval_seed_to_docs(payload: dict[str, Any]) -> list[dict[str, str]]:
    if not isinstance(payload, dict):
        return []

    docs: list[dict[str, str]] = []

    def _push_list(key: str, source: str, limit: int = 6) -> None:
        value = payload.get(key, [])
        if not isinstance(value, list):
            return
        cleaned = [str(x).strip() for x in value if str(x).strip()]
        if cleaned:
            docs.append({"source": source, "text": "\n".join(cleaned[:limit])})

    _push_list("retrieval_queries", "gpt_seed # queries", limit=8)
    _push_list("retrieval_hints", "gpt_seed # hints", limit=8)
    _push_list("visual_facts", "gpt_seed # visual_facts", limit=8)
    _push_list("option_discriminators", "gpt_seed # option_discriminators", limit=8)
    _push_list("avoid_topics", "gpt_seed # avoid_topics", limit=8)

    return docs[: int(getattr(settings, "gpt_retrieval_seed_max_docs", 6) or 6)]


def _merge_retrieval_docs(base_docs: list[dict[str, str]], seed_docs: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for d in list(base_docs or []) + list(seed_docs or []):
        if not isinstance(d, dict):
            continue
        src = str(d.get("source", "unknown")).strip()
        txt = str(d.get("text", "")).strip()
        if not txt:
            continue
        key = (src, txt)
        if key in seen:
            continue
        seen.add(key)
        out.append({"source": src, "text": txt})

    return out


def augment_retrieval_docs_with_gpt(
    *,
    llm: Any,
    sample: dict[str, Any],
    image_ref: str | None,
    retrieval_docs: list[dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    trace: dict[str, Any] = {
        "enabled": bool(getattr(settings, "enable_gpt_retrieval_seed", False)),
        "called": False,
        "ok": False,
        "docs_added": 0,
        "model": str(getattr(settings, "gpt_retrieval_seed_model", "gpt-5.1") or "gpt-5.1"),
        "error": "",
    }

    if not trace["enabled"] or llm is None:
        return retrieval_docs, trace

    question = str(sample.get("question", "")).strip()
    choices = _to_choice_list(sample.get("choices"))
    dataset_type = str(sample.get("dataset_type", "generic")).strip().lower()
    image_data_url = _image_ref_to_data_url_for_retrieval_seed(llm, image_ref)

    trace["called"] = True
    result = _call_xiaoai_gpt_retrieval_seed(
        question=question,
        choices=choices,
        dataset_type=dataset_type,
        image_data_url=image_data_url,
        llm=llm,
    )

    if not bool(result.get("ok", False)):
        trace["error"] = str(result.get("error", "unknown_error"))
        return retrieval_docs, trace

    seed_docs = _parse_gpt_retrieval_seed_to_docs(result.get("data", {}))
    merged = _merge_retrieval_docs(retrieval_docs, seed_docs)

    trace["ok"] = True
    trace["docs_added"] = max(0, len(merged) - len(retrieval_docs))
    return merged, trace


def get_fallback_docs(
    sample: dict[str, Any],
    retrieval_docs: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    out = list(get_baseline_docs(sample))
    seen = {(str(d.get("source", "")), str(d.get("text", ""))) for d in out}

    for d in list(retrieval_docs or []):
        if not isinstance(d, dict):
            continue
        src = str(d.get("source", "")).strip()
        txt = str(d.get("text", "")).strip()
        if not txt:
            continue
        if not (
            src.startswith("gpt_seed # visual_facts")
            or src.startswith("gpt_seed # option_discriminators")
            or src.startswith("local_image")
            or src.startswith("seed_local_image")
        ):
            continue
        key = (src, txt)
        if key in seen:
            continue
        seen.add(key)
        out.append({"source": src, "text": txt})

    return out


def _to_text_list(v: Any) -> list[str]:
    if v is None or _is_missing(v):
        return []

    if hasattr(v, "tolist"):
        try:
            v = v.tolist()
        except Exception:
            pass

    if isinstance(v, (list, tuple, set)):
        out = []
        for x in v:
            if _is_missing(x):
                continue
            s = str(x).strip()
            if s:
                out.append(s)
        return out

    if isinstance(v, str):
        s = v.strip()
        if not s:
            return []
        try:
            parsed = ast.literal_eval(s)
        except (SyntaxError, ValueError):
            return [s]
        if isinstance(parsed, (list, tuple, set)):
            out = []
            for x in parsed:
                if _is_missing(x):
                    continue
                sx = str(x).strip()
                if sx:
                    out.append(sx)
            return out
        sp = str(parsed).strip()
        return [sp] if sp else []

    s = str(v).strip()
    return [s] if s else []


def _is_placeholder_image_text(v: str) -> bool:
    t = v.strip().lower()
    return t in {"?", "not supported with pagination yet", "none", "null", "nan", ""}


def _decode_choice_answer(answer_value: Any, choices: list[str]) -> tuple[str, str | None, int | None]:
    answer_raw = "" if _is_missing(answer_value) else str(answer_value).strip()
    if not answer_raw:
        return "", None, None

    upper = answer_raw.upper().strip()
    if len(upper) == 1 and "A" <= upper <= "Z":
        idx = ord(upper) - ord("A")
        if 0 <= idx < len(choices):
            return choices[idx], upper, idx
        return answer_raw, upper, None

    if upper.startswith("(") and upper.endswith(")") and len(upper) == 3:
        ch = upper[1]
        if "A" <= ch <= "Z":
            idx = ord(ch) - ord("A")
            if 0 <= idx < len(choices):
                return choices[idx], ch, idx

    return answer_raw, None, None


def _norm_choice_text(text: str) -> str:
    s = str(text or "").strip().lower()
    s = re.sub(r"^[\(\[]?[a-zA-Z][\)\].:\-]\s*", "", s)
    s = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", s)
    return s


def _map_answer_to_choices(answer: str, choices: list[str]) -> str:
    raw = str(answer or "").strip()
    if not raw or not choices:
        return raw

    # 先将现有的字母/括号字母解析
    decoded_text, _, decoded_idx = _decode_choice_answer(raw, choices)
    if decoded_idx is not None and 0 <= decoded_idx < len(choices):
        return str(choices[decoded_idx])

    raw_norm = _norm_choice_text(raw)
    if not raw_norm:
        return raw

    # 精确 / 包含匹配
    best_choice = None
    best_score = -1.0

    for ch in choices:
        ch_text = str(ch)
        ch_norm = _norm_choice_text(ch_text)
        if not ch_norm:
            continue

        if raw_norm == ch_norm:
            return ch_text
        if raw_norm in ch_norm or ch_norm in raw_norm:
            score = min(len(raw_norm), len(ch_norm)) / max(len(raw_norm), len(ch_norm), 1)
            if score > best_score:
                best_score = score
                best_choice = ch_text

    if best_choice is not None and best_score >= 0.45:
        return best_choice

    # 词面 token overlap
    raw_toks = set(re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", raw.lower()))
    raw_toks = {t for t in raw_toks if len(t) >= 2}
    if raw_toks:
        for ch in choices:
            ch_text = str(ch)
            ch_toks = set(re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", ch_text.lower()))
            ch_toks = {t for t in ch_toks if len(t) >= 2}
            if not ch_toks:
                continue
            overlap = len(raw_toks & ch_toks) / max(len(raw_toks | ch_toks), 1)
            if overlap > best_score:
                best_score = overlap
                best_choice = ch_text

    if best_choice is not None and best_score >= 0.34:
        return best_choice

    return raw


def _extract_mmmu_choices(record: dict[str, Any]) -> list[str]:
    if "choices" in record:
        choices = _to_choice_list(record.get("choices"))
        if choices:
            return choices
    if "options" in record:
        options = _to_choice_list(record.get("options"))
        if options:
            return options

    keyed: list[str] = []
    for k in [
        "option_a",
        "option_b",
        "option_c",
        "option_d",
        "option_e",
        "option_f",
        "option_g",
        "option_h",
        "option_i",
        "option_j",
    ]:
        v = record.get(k)
        if _is_missing(v):
            continue
        keyed.append(str(v))
    return keyed


def _normalize_aokvqa_record(record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    q = record.get(args.question_field)
    if _is_missing(q):
        raise ValueError(f"Missing question field: {args.question_field}")

    choices = _to_choice_list(record.get("choices"))
    if not choices:
        choices = _to_choice_list(record.get("options"))

    answer_text = ""
    answer_label: str | None = None
    answer_idx: int | None = None

    # 先解析 direct_answers，但不优先把它直接当最终 gold
    direct_answers = _to_text_list(record.get("direct_answers"))

    # AOKVQA 当前整条推理/评测链是按选择题选项文本来跑的，
    idx = record.get("correct_choice_idx")
    idx_valid = False
    if idx is not None:
        if isinstance(idx, (int, float)):
            idx_valid = not (isinstance(idx, float) and math.isnan(idx))
        else:
            try:
                idx_float = float(idx)
                idx_valid = not math.isnan(idx_float)
            except Exception:
                idx_valid = False

    if idx_valid and choices:
        try:
            i_val = int(idx)
            if 0 <= i_val < len(choices):
                answer_idx = i_val
                answer_label = chr(ord("A") + i_val)
                answer_text = str(choices[i_val])
        except Exception:
            pass

    # 如果没有 correct_choice_idx，再尝试 answer_field
    if not answer_text:
        answer_value = record.get(args.answer_field)
        answer_text, answer_label, answer_idx = _decode_choice_answer(answer_value, choices)

    # 再不行才退回 direct_answers 的众数
    if not answer_text and direct_answers:
        answer_text = str(Counter(direct_answers).most_common(1)[0][0])

        # 如果众数能匹配到某个选项，就对齐成标准选项文本
        if choices:
            matched_idx = _match_choice_text(answer_text, choices)
            if matched_idx is not None:
                answer_idx = matched_idx
                answer_label = chr(ord("A") + matched_idx)
                answer_text = str(choices[matched_idx])

    sample: dict[str, Any] = {
        "question": str(q),
        "answer": answer_text,
        "choices": choices,
        "dataset_type": "aokvqa",
    }

    if answer_label is not None:
        sample["answer_label"] = answer_label
    if answer_idx is not None:
        sample["correct_choice_idx"] = answer_idx

    image_candidates: list[dict[str, Any]] = []
    for key in [args.image_url_field, args.image_path_field, "image", "img_path"]:
        if not key or key not in record:
            continue
        value = record.get(key)
        if _is_missing(value) or value == "":
            continue

        if key == "image" and isinstance(value, dict) and isinstance(value.get("bytes"), (bytes, bytearray)):
            sample["__image_bytes"] = bytes(value.get("bytes"))
            sample["__image_ext"] = ".jpg"
            continue

        if isinstance(value, str):
            if value.startswith("http://") or value.startswith("https://"):
                image_candidates.append({"url": value, "source": key})
            else:
                image_candidates.append({"path": value, "source": key})

    if image_candidates:
        sample["__image_candidates"] = image_candidates
        first = image_candidates[0]
        if "url" in first:
            sample["image_url"] = first["url"]
        elif "path" in first:
            sample["image_path"] = first["path"]

    docs: list[dict[str, str]] = []
    docs_value = record.get(args.documents_field) if args.documents_field else None
    if isinstance(docs_value, list):
        docs = docs_value
    elif isinstance(docs_value, str) and docs_value.strip():
        docs = [{"source": args.documents_field or "context", "text": docs_value}]

    # 兼容单条 rationale
    rationale = record.get("rationale")
    if isinstance(rationale, str) and rationale.strip():
        docs.append({"source": "rationale", "text": rationale.strip()})

    # 兼容 AOKVQA 源文件里的 rationales 列表
    rationales = record.get("rationales")
    rationales_list = _to_text_list(rationales)
    if rationales_list:
        docs.append({"source": "rationale", "text": "\n".join(rationales_list)})

    sample["documents"] = docs

    # 保留原字段，便于分析
    for k in [
        "id",
        "question_id",
        "image_id",
        "difficult",
        "direct_answers",
        "difficult_direct_answer",
        "distractors",
        "domain",
        "category",
        "subdomain",
        "topic",
    ]:
        if k in record:
            sample[k] = record[k]

    sample["__original_record"] = _jsonable(record)
    return sample


def _normalize_mmmu_record(record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    q = record.get(args.question_field)
    if _is_missing(q):
        raise ValueError(f"Missing question field: {args.question_field}")

    choices = _extract_mmmu_choices(record)
    answer_text, answer_label, answer_idx = _decode_choice_answer(record.get(args.answer_field), choices)
    if not answer_text:
        answer_text = _extract_answer(record, args)

    sample: dict[str, Any] = {
        "question": str(q),
        "answer": answer_text,
        "choices": choices,
        "dataset_type": "mmmu_pro",
    }
    if answer_label is not None:
        sample["answer_label"] = answer_label
    if answer_idx is not None:
        sample["correct_choice_idx"] = answer_idx

    image_candidates: list[dict[str, Any]] = []

    def _push_image_candidate(value: Any, source_key: str) -> None:
        if _is_missing(value):
            return
        if isinstance(value, dict) and isinstance(value.get("bytes"), (bytes, bytearray)):
            image_candidates.append(
                {
                    "bytes": bytes(value.get("bytes")),
                    "ext": Path(str(value.get("path", ".jpg"))).suffix or ".jpg",
                    "source": source_key,
                }
            )
            return
        if isinstance(value, str):
            if _is_placeholder_image_text(value):
                return
            vv = value.strip()
            if vv.startswith("http://") or vv.startswith("https://"):
                image_candidates.append({"url": vv, "source": source_key})
            else:
                image_candidates.append({"path": vv, "source": source_key})

    for key in [
        args.image_url_field,
        args.image_path_field,
        "image",
        "img_path",
        "image_1",
        "image_2",
        "image_3",
        "image_4",
        "image_5",
        "image_6",
        "image_7",
    ]:
        if not key or key not in record:
            continue
        _push_image_candidate(record.get(key), key)

    if image_candidates:
        sample["__image_candidates"] = image_candidates
        first = image_candidates[0]
        if "bytes" in first:
            sample["__image_bytes"] = first["bytes"]
            sample["__image_ext"] = first.get("ext", ".jpg")
        elif "url" in first:
            sample["image_url"] = first["url"]
        elif "path" in first:
            sample["image_path"] = first["path"]

    docs: list[dict[str, str]] = []
    for k in ["context", "hint", "rationale", "solution", "explanation"]:
        v = record.get(k)
        if isinstance(v, str) and v.strip() and not _is_placeholder_image_text(v):
            docs.append({"source": k, "text": v})

    if not docs:
        docs_value = record.get(args.documents_field) if args.documents_field else None
        if isinstance(docs_value, list):
            docs = docs_value
        elif isinstance(docs_value, str) and docs_value.strip():
            docs = [{"source": args.documents_field or "context", "text": docs_value}]
    sample["documents"] = docs

    for k in ["id", "question_id", "image_id", "domain", "category", "subdomain", "topic", "options", "has_table", "has_chart"]:
        if k in record:
            sample[k] = record[k]

    sample["__original_record"] = _jsonable(record)
    return sample


def _normalize_m3cot_record(record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    q = record.get(args.question_field)
    if _is_missing(q):
        raise ValueError(f"Missing question field: {args.question_field}")

    choices = _to_choice_list(record.get("choices"))
    answer_text, answer_label, answer_idx = _decode_choice_answer(record.get(args.answer_field), choices)
    if not answer_text:
        answer_text = _extract_answer(record, args)

    sample: dict[str, Any] = {
        "question": str(q),
        "answer": answer_text,
        "choices": choices,
        "dataset_type": "m3cot",
    }
    if answer_label is not None:
        sample["answer_label"] = answer_label
    if answer_idx is not None:
        sample["correct_choice_idx"] = answer_idx

    for key in [args.image_url_field, args.image_path_field, "image", "img_path"]:
        if not key or key not in record:
            continue
        value = record.get(key)
        if _is_missing(value) or value == "":
            continue
        sample[key] = value

    docs: list[dict[str, str]] = []

    context = record.get("context")
    if isinstance(context, str) and context.strip():
        docs.append({"source": "context", "text": context.strip()})

    rationale = record.get("rationale")
    if isinstance(rationale, str) and rationale.strip():
        docs.append({"source": "rationale", "text": rationale.strip()})

    explain = record.get("explain")
    if isinstance(explain, str) and explain.strip():
        docs.append({"source": "explanation", "text": explain.strip()})

    explanation = record.get("explanation")
    if isinstance(explanation, str) and explanation.strip():
        docs.append({"source": "explanation", "text": explanation.strip()})

    if not docs:
        docs_value = record.get(args.documents_field) if args.documents_field else None
        if isinstance(docs_value, list):
            docs = docs_value
        elif isinstance(docs_value, str) and docs_value.strip():
            docs = [{"source": args.documents_field or "context", "text": docs_value.strip()}]

    sample["documents"] = docs

    for k in ["reasoning_hops", "cot_steps", "steps", "domain", "options", "has_table", "has_chart", "topic", "id", "category"]:
        if k in record:
            sample[k] = record[k]

    sample["__original_record"] = _jsonable(record)
    return sample


def _normalize_cmmqa_record(record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    q = record.get(args.question_field)
    if _is_missing(q):
        raise ValueError(f"Missing question field: {args.question_field}")

    choices_raw = record.get("choices")
    if _is_missing(choices_raw):
        choices_raw = record.get("options")
    choices = _to_choice_list(choices_raw)
    answer_text, answer_label, answer_idx = _decode_choice_answer(record.get(args.answer_field), choices)
    if not answer_text:
        answer_text = _extract_answer(record, args)

    sample: dict[str, Any] = {
        "question": str(q),
        "answer": answer_text,
        "choices": choices,
        "dataset_type": "cmmqa",
    }
    if answer_label is not None:
        sample["answer_label"] = answer_label
    if answer_idx is not None:
        sample["correct_choice_idx"] = answer_idx

    for key in [args.image_url_field, args.image_path_field, "image", "img_path", "image_url"]:
        if not key or key not in record:
            continue
        value = record.get(key)
        if _is_missing(value) or value == "":
            continue
        sample[key] = value

    docs: list[dict[str, str]] = []

    reasoning_chains = record.get("reasoning_chains")
    if isinstance(reasoning_chains, list) and reasoning_chains:
        chain_lines: list[str] = []
        for idx, chain in enumerate(reasoning_chains, start=1):
            if isinstance(chain, (list, tuple)) and len(chain) >= 3:
                chain_lines.append(f"{idx}. {chain[0]} --{chain[1]}--> {chain[2]}")
            elif isinstance(chain, str) and chain.strip():
                chain_lines.append(f"{idx}. {chain.strip()}")
        if chain_lines:
            docs.append({"source": "reasoning_chains", "text": "\n".join(chain_lines)})

    rationale = record.get("rationale")
    if isinstance(rationale, str) and rationale.strip():
        docs.append({"source": "rationale", "text": rationale.strip()})

    explain = record.get("explain")
    if isinstance(explain, str) and explain.strip():
        docs.append({"source": "explanation", "text": explain.strip()})

    explanation = record.get("explanation")
    if isinstance(explanation, str) and explanation.strip():
        docs.append({"source": "explanation", "text": explanation.strip()})

    sample["documents"] = docs

    for k in ["ID", "id", "question_id", "image_id", "image", "image_url", "reasoning_chains"]:
        if k in record:
            sample[k] = record[k]

    sample["__original_record"] = _jsonable(record)
    return sample


def _normalize_scienceqa_record(record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    q = record.get(args.question_field)
    if _is_missing(q):
        raise ValueError(f"Missing question field: {args.question_field}")

    choices_raw = record.get("choices")
    if _is_missing(choices_raw):
        choices_raw = record.get("options")
    choices = _to_choice_list(choices_raw)

    answer_value = record.get(args.answer_field)
    answer_text = ""
    answer_label: str | None = None
    answer_idx: int | None = None

    if isinstance(answer_value, int) and 0 <= answer_value < len(choices):
        answer_idx = int(answer_value)
        answer_label = chr(ord("A") + answer_idx)
        answer_text = choices[answer_idx]
    else:
        answer_text, answer_label, answer_idx = _decode_choice_answer(answer_value, choices)

    if not answer_text:
        answer_text = _extract_answer(record, args)

    sample: dict[str, Any] = {
        "question": str(q),
        "answer": answer_text,
        "choices": choices,
        "dataset_type": "scienceqa",
    }
    if answer_label is not None:
        sample["answer_label"] = answer_label
    if answer_idx is not None:
        sample["correct_choice_idx"] = answer_idx

    rid = None
    for key in ["id", "ID", "question_id", "__record_id"]:
        v = record.get(key)
        if _is_missing(v):
            continue
        rid = v
        break

    split_value = record.get("split")
    split = "" if _is_missing(split_value) else str(split_value)

    image_name = record.get("image")
    if isinstance(image_name, str) and image_name.strip():
        if split and rid is not None:
            sample["image_path"] = f"{split}/{rid}/{image_name.strip()}"
        elif rid is not None:
            sample["image_path"] = f"{rid}/{image_name.strip()}"
        else:
            sample["image_path"] = image_name.strip()

    # 这里只保留原始文档型字段
    docs: list[dict[str, str]] = []
    for k in ["hint", "lecture", "solution"]:
        v = record.get(k)
        if isinstance(v, str) and v.strip():
            docs.append({"source": k, "text": v.strip()})
    sample["documents"] = docs

    for k in ["id", "ID", "split", "task", "grade", "subject", "topic", "category", "skill", "image"]:
        if k in record:
            sample[k] = record[k]

    sample["__original_record"] = _jsonable(record)
    return sample


def _infer_dataset_type(args: argparse.Namespace, record: dict[str, Any], dataset_path: Path) -> str:
    if args.dataset_type != "auto":
        return args.dataset_type

    path_l = str(dataset_path).lower()

    # 先看显式路径
    if "aokvqa" in path_l or "a_okvqa" in path_l:
        return "aokvqa"
    if "scienceqa" in path_l:
        return "scienceqa"
    if "cmmqa" in path_l:
        return "cmmqa"
    if "mmmu" in path_l:
        return "mmmu_pro"
    if "m3cot" in path_l:
        return "m3cot"

    # 再看显式字段
    if "direct_answers" in record:
        return "aokvqa"

    # 再走结构启发式
    if {"question_id", "image_id"}.intersection(record.keys()) and (
        "options" in record or "choices" in record or "option_a" in record
    ):
        return "mmmu_pro"

    if {"rationale", "topic", "category", "image_id"}.intersection(record.keys()) and "choices" in record:
        return "m3cot"

    return "generic"


def _normalize_sample_record(record: dict[str, Any], args: argparse.Namespace, dataset_type: str = "generic") -> dict[str, Any]:
    if dataset_type == "aokvqa":
        return _normalize_aokvqa_record(record, args)
    if dataset_type == "mmmu_pro":
        return _normalize_mmmu_record(record, args)
    if dataset_type == "m3cot":
        return _normalize_m3cot_record(record, args)
    if dataset_type == "cmmqa":
        return _normalize_cmmqa_record(record, args)
    if dataset_type == "scienceqa":
        return _normalize_scienceqa_record(record, args)

    q = record.get(args.question_field)
    if _is_missing(q):
        raise ValueError(f"Missing question field: {args.question_field}")

    sample: dict[str, Any] = {"question": str(q), "answer": _extract_answer(record, args)}

    for key in [args.image_url_field, args.image_path_field, "image", "img_path"]:
        if not key or key not in record:
            continue
        value = record.get(key)
        if _is_missing(value) or value == "":
            continue
        if key == "image" and isinstance(value, dict) and isinstance(value.get("bytes"), (bytes, bytearray)):
            sample["__image_bytes"] = bytes(value.get("bytes"))
            sample["__image_ext"] = ".jpg"
            continue
        sample[key] = value

    docs: list[dict[str, str]] = []

    docs_value = record.get(args.documents_field) if args.documents_field else None
    if isinstance(docs_value, list):
        docs = docs_value
    elif isinstance(docs_value, str) and docs_value.strip():
        docs = [{"source": args.documents_field or "context", "text": docs_value.strip()}]

    rationale = record.get("rationale")
    if isinstance(rationale, str) and rationale.strip():
        docs.append({"source": "rationale", "text": rationale.strip()})

    rationales = _to_text_list(record.get("rationales"))
    if rationales:
        docs.append({"source": "rationale", "text": "\n".join(rationales)})

    explain = record.get("explain")
    if isinstance(explain, str) and explain.strip():
        docs.append({"source": "explanation", "text": explain.strip()})

    explanation = record.get("explanation")
    if isinstance(explanation, str) and explanation.strip():
        docs.append({"source": "explanation", "text": explanation.strip()})

    sample["documents"] = docs

    for k in ["reasoning_hops", "cot_steps", "steps", "domain", "choices", "options", "has_table", "has_chart"]:
        if k in record:
            sample[k] = record[k]

    sample["dataset_type"] = "generic"
    sample["__original_record"] = _jsonable(record)
    return sample


def load_dataset(dataset_path: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    if dataset_path.is_dir():
        parquet_files = sorted(dataset_path.glob("*.parquet"))
        if parquet_files:
            merged_records: list[dict[str, Any]] = []
            for pf in parquet_files:
                merged_records.extend(load_dataset(pf, args))
            return merged_records

        problems_json = dataset_path / "problems.json"
        if problems_json.exists():
            return load_dataset(problems_json, args)

        json_files = sorted(dataset_path.glob("*.json"))
        if len(json_files) == 1:
            return load_dataset(json_files[0], args)

        raise ValueError(f"Unsupported dataset directory format: {dataset_path}")

    suffix = dataset_path.suffix.lower()

    def _normalize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        vision_map = _load_mmmu_vision_map(args)
        out: list[dict[str, Any]] = []
        for x in records:
            if not isinstance(x, dict):
                continue
            ds_type = _infer_dataset_type(args, x, dataset_path)
            record = dict(x)
            if ds_type == "mmmu_pro":
                merged = _merge_mmmu_vision_record(record, vision_map, args)
                out.append(_normalize_sample_record(merged, args, dataset_type=ds_type))
            else:
                out.append(_normalize_sample_record(record, args, dataset_type=ds_type))
        return out

    if suffix == ".json":
        raw = json.loads(dataset_path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return _normalize_records(raw)
        if isinstance(raw, dict):
            unpacked: list[dict[str, Any]] = []
            for k, v in raw.items():
                if isinstance(v, dict):
                    item = dict(v)
                    item.setdefault("__record_id", str(k))
                    item.setdefault("id", str(k))
                    unpacked.append(item)
            if unpacked:
                return _normalize_records(unpacked)
        raise ValueError("JSON dataset must be a list of objects or a dict[id->object].")

    if suffix == ".jsonl":
        out: list[dict[str, Any]] = []
        for line in dataset_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                out.append(obj)
        return _normalize_records(out)

    if suffix == ".parquet":
        try:
            import pandas as pd
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Reading parquet datasets requires pandas (and pyarrow/fastparquet). Please install requirements.txt first."
            ) from exc
        df = pd.read_parquet(dataset_path)
        records = df.to_dict(orient="records")
        return _normalize_records(records)

    raise ValueError(f"Unsupported dataset format: {suffix}. Use .json/.jsonl/.parquet")


def _vision_record_id(record: dict[str, Any], key_hint: str) -> str | None:
    candidates = [key_hint, "id", "question_id", "image_id", "sample_id"]
    for k in candidates:
        if not k:
            continue
        v = record.get(k)
        if _is_missing(v):
            continue
        s = str(v).strip()
        if s:
            return s
    return None


def _load_mmmu_vision_map(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    vision_path = args.mmmu_vision_parquet
    if vision_path is None or not vision_path.exists():
        return {}
    try:
        import pandas as pd
    except ModuleNotFoundError:
        return {}

    vision_files: list[Path]
    if vision_path.is_dir():
        vision_files = sorted(vision_path.glob("*.parquet"))
    else:
        vision_files = [vision_path]
    if not vision_files:
        return {}

    out: dict[str, dict[str, Any]] = {}
    for vf in vision_files:
        records = pd.read_parquet(vf).to_dict(orient="records")
        for rec in records:
            if not isinstance(rec, dict):
                continue
            rid = _vision_record_id(rec, args.mmmu_vision_id_field)
            if rid:
                out[rid] = rec
    return out


def _merge_mmmu_vision_record(record: dict[str, Any], vision_map: dict[str, dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    if not vision_map:
        return record
    rid = _vision_record_id(record, args.mmmu_id_field)
    if not rid:
        return record
    vision = vision_map.get(rid)
    if not vision:
        return record

    merged = dict(record)
    image_field = args.mmmu_vision_image_field
    if image_field in vision and "image" not in merged and "image_path" not in merged and "img_path" not in merged:
        merged["image"] = vision.get(image_field)
    for k in ["image_1", "image_2", "image_3", "image_4", "image_5", "image_6", "image_7"]:
        if k in vision and k not in merged:
            merged[k] = vision.get(k)
    if "image_path" in vision and "image_path" not in merged:
        merged["image_path"] = vision.get("image_path")
    if "image_url" in vision and "image_url" not in merged:
        merged["image_url"] = vision.get("image_url")
    return merged

def _prefer_local_image_first(sample: dict[str, Any]) -> bool:
    """除 CMMQA 外，其余数据集优先走本地图片路径。"""
    dataset_type = str(sample.get("dataset_type", "generic")).lower()
    return dataset_type != "cmmqa"


def _resolve_local_image_path(candidate: str, image_root: Path | None = None) -> Path | None:
    raw = candidate.strip()
    if not raw or _is_placeholder_image_text(raw):
        return None

    norm = raw.replace("\\", "/")
    base = Path(norm)
    path_candidates: list[Path] = []

    # 原始候选
    path_candidates.append(base)

    if image_root is not None:
        # 常规拼接
        path_candidates.append(image_root / base)
        path_candidates.append(image_root / base.name)

        # 常见 images 子目录兼容
        path_candidates.append(image_root / "images" / base)
        path_candidates.append(image_root / "images" / base.name)

        # data/images 前缀剥离兼容
        parts = base.parts
        if len(parts) >= 2 and parts[0].lower() == "data" and parts[1].lower() == "images":
            stripped = Path(*parts[2:])
            path_candidates.append(image_root / stripped)
            path_candidates.append(image_root / stripped.name)

    seen: set[Path] = set()
    for p in path_candidates:
        try:
            rp = p.expanduser().resolve()
        except Exception:
            continue
        if rp in seen:
            continue
        seen.add(rp)
        if rp.exists() and rp.is_file():
            return rp

    return None


def _extract_candidate_lists(sample: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    candidates = sample.get("__image_candidates")
    structured: list[dict[str, Any]] = []
    if isinstance(candidates, list):
        structured = [c for c in candidates if isinstance(c, dict)]

    urls: list[str] = []
    paths: list[str] = []

    for c in structured:
        if isinstance(c.get("url"), str) and c.get("url", "").strip():
            urls.append(c["url"].strip())
        if isinstance(c.get("path"), str) and c.get("path", "").strip():
            paths.append(c["path"].strip())

    for key in ["image_url"]:
        v = sample.get(key)
        if isinstance(v, str) and v.strip():
            urls.append(v.strip())

    for key in ["image_path", "image", "img_path"]:
        v = sample.get(key)
        if isinstance(v, str) and v.strip():
            paths.append(v.strip())

    return structured, urls, paths

def _looks_like_inline_image_ref(image_ref: str | None) -> bool:
    """
    现在视觉输入统一要求走 vLLM 可访问的本地文件 / file://。
    因此这里只检测真正的 data URL；不再用超长字符串猜 inline image。
    """
    s = str(image_ref or "").strip().lower()
    return s.startswith("data:image/")

def resolve_image_reference(
    sample: dict[str, Any],
    image_root: Path | None = None,
    vllm_local_media_root: Path | None = None,
) -> tuple[str | None, str | None]:
    prefer_local = _prefer_local_image_first(sample)
    structured, urls, paths = _extract_candidate_lists(sample)
    bucket = _dataset_media_bucket(sample)

    def _require_media_root() -> bool:
        return vllm_local_media_root is not None

    def _try_bytes() -> tuple[str | None, str | None]:
        for c in structured:
            if isinstance(c.get("bytes"), (bytes, bytearray)):
                if not _require_media_root():
                    return None, None
                ext = str(c.get("ext", ".jpg"))
                ref = _persist_image_bytes_to_vllm_media_root(
                    bytes(c["bytes"]),
                    media_root=vllm_local_media_root,
                    bucket=bucket,
                    ext=ext,
                )
                return ref, str(c.get("source", "image_bytes"))

        img_bytes = sample.get("__image_bytes")
        if isinstance(img_bytes, (bytes, bytearray)):
            if not _require_media_root():
                return None, None
            ext = str(sample.get("__image_ext", ".jpg"))
            ref = _persist_image_bytes_to_vllm_media_root(
                bytes(img_bytes),
                media_root=vllm_local_media_root,
                bucket=bucket,
                ext=ext,
            )
            return ref, "image_bytes"

        return None, None

    def _try_paths() -> tuple[str | None, str | None]:
        for p in paths:
            rp = _resolve_local_image_path(p, image_root=image_root)
            if rp is not None:
                if not _require_media_root():
                    return None, None
                return _stage_local_image_to_vllm_media_root(
                    rp,
                    media_root=vllm_local_media_root,
                    bucket=bucket,
                ), "image_path"
        return None, None

    def _try_urls() -> tuple[str | None, str | None]:
        for u in urls:
            if isinstance(u, str) and u.strip():
                return u.strip(), "image_url"
        return None, None

    ref, src = _try_bytes()
    if ref:
        return ref, src

    if prefer_local:
        ref, src = _try_paths()
        if ref:
            return ref, src
        ref, src = _try_urls()
        if ref:
            return ref, src
    else:
        ref, src = _try_urls()
        if ref:
            return ref, src
        ref, src = _try_paths()
        if ref:
            return ref, src

    return None, None


def _clip_prompt_text(text: str, max_chars: int) -> str:
    s = str(text or "").strip()
    if max_chars <= 0:
        return ""
    return s if len(s) <= max_chars else s[:max_chars]


def _head_tail_clip(text: str, max_chars: int = 1600, head_chars: int = 1000) -> str:
    s = str(text or "").strip()
    if len(s) <= max_chars:
        return s

    head_chars = max(200, min(head_chars, max_chars - 200))
    tail_chars = max_chars - head_chars - 20
    if tail_chars <= 0:
        return s[:max_chars]

    return s[:head_chars] + "\n...\n<truncated>\n...\n" + s[-tail_chars:]


def answer_with_context(
    llm: Any,
    question: str,
    context: str,
    image_ref: str | None = None,
    dataset_type: str = "generic",
    choices: list[str] | None = None,
) -> str:
    if llm is None:
        return context[:80] or "insufficient_context"

    choices = choices or []

    if choices:
        opts = []
        max_opts = min(len(choices), 10)
        for i in range(max_opts):
            label = chr(ord("A") + i)
            opts.append(f"{label}. {choices[i]}")
        option_text = "\n".join(opts)

        dataset_hint = {
            "cmmqa": "先识别图像关键实体，再结合知识链做约束推理。",
            "scienceqa": "先定位科学概念与图像线索，再结合 hint/lecture 做约束推理。",
            "mmmu_pro": "结合图文线索完成多模态选择题判断。",
        }.get(dataset_type, "请严格在给定选项中选择最符合题意的一项。")

        prompt = (
            "假设你是专业的多模态 VQA 选择题评测专家。"
            f"{dataset_hint}"
            "仅输出一个选项字母，不要输出解释。\n"
            f"题目: {question}\n"
            f"选项:\n{option_text}\n"
            f"辅助上下文:\n{context[:1600]}"
        )
    else:
        prompt = (
            "假设你是专业的多模态 VQA 专家。基于给定上下文回答问题，"
            "若上下文不足则明确说明。只输出简洁答案。\n"
            f"问题: {question}\n上下文:\n{context[:4000]}"
        )

    if image_ref:
        raw = llm.chat_with_image(prompt=prompt, image_url=image_ref, temperature=0.1)
    else:
        raw = llm.chat(
            [{"role": "system", "content": "假设你是一名专业的 VQA 推理专家"}, {"role": "user", "content": prompt}],
            temperature=0.1,
        )

    return _map_answer_to_choices(raw, choices) if choices else raw


def refine_with_sft(
    llm: Any,
    question: str,
    first_answer: str,
    context: str,
    image_ref: str | None = None,
    dataset_type: str = "generic",
    choices: list[str] | None = None,
) -> str:
    if llm is None:
        return first_answer

    q_text = _clip_prompt_text(question, 800)
    ans_text = _clip_prompt_text(first_answer, 500)
    ctx_text = _clip_prompt_text(context, 1200)

    opts = []
    if choices:
        for i, ch in enumerate(choices[:10]):
            opts.append(f"{chr(ord('A') + i)}. {ch}")

    answer_constraint = "如果是选择题，仅输出选项字母。\n"
    if dataset_type == "aokvqa" and choices:
        answer_constraint = "如果是 AOKVQA 选择题，仅输出与选项完全一致的选项文本，不要输出字母和解释。\n"
    elif dataset_type == "m3cot" and choices:
        answer_constraint = (
            "如果是 M3CoT 选择题，请先根据图像比较关系完成自校正，再仅输出与选项完全一致的选项文本，"
            "不要输出字母，不要输出解释。\n"
        )

    extra_instruction = ""
    if dataset_type == "m3cot":
        extra_instruction = (
            "自校正时请重点检查：\n"
            "1) 是否真的比较了图中被对比的两组对象；\n"
            "2) 是否把 Pair 1 / Pair 2、左 / 右、前 / 后混淆；\n"
            "3) 是否先从图像中得到距离、位置或方向关系，再应用题目规则；\n"
            "4) 如果首轮答案只是泛化描述或没有明确落到选项，请纠正为唯一选项文本。\n"
        )

    prompt = (
        "假设你是一名专业的 VQA 推理助手。请结合首轮答案和证据进行一次自校正，仅输出最终答案。\n"
        f"{answer_constraint}"
        f"{extra_instruction}"
        f"题目: {q_text}\n"
        f"首轮答案: {ans_text}\n"
        f"选项:\n{chr(10).join(opts)}\n"
        f"证据上下文:\n{ctx_text}\n"
        f"数据集类型: {dataset_type}"
    )

    image_ref = _require_vllm_compatible_image_ref(image_ref)

    if image_ref:
        raw = llm.chat_with_image(prompt=prompt, image_url=image_ref, temperature=0.1)
    else:
        raw = llm.chat(
            [{"role": "system", "content": "假设你是一个专业的 VQA 评测专家"}, {"role": "user", "content": prompt}],
            temperature=0.1,
        )

    return _map_answer_to_choices(raw, choices or []) if choices else raw


class ExternalKnowledgeBase:
    def __init__(self):
        self.knowledge_cache: dict[str, list[dict]] = {}
        self.common_sense: dict[str, str] = self._load_common_sense()

    def _load_common_sense(self) -> dict[str, str]:
        return {
            "airplane": "Aircraft used for air travel",
            "fence": "Barrier structure for enclosing areas",
            "giraffe": "Tallest terrestrial animal with long neck",
            "elephant": "Large mammal with trunk and tusks",
            "soccer": "Team sport played with a ball",
            "football": "Sport involving kicking a ball",
            "skateboarding": "Sport using a board with wheels",
        }

    def retrieve(self, question: str, choices: list[str] | None = None) -> list[dict[str, str]]:
        docs: list[dict[str, str]] = []
        q_lower = question.lower()

        for keyword, knowledge in self.common_sense.items():
            if keyword in q_lower:
                docs.append({"source": "common_sense", "text": f"{keyword}: {knowledge}"})

        if "what" in q_lower or "什么" in question:
            docs.append({"source": "question_type", "text": "This is a 'what' question requiring object identification"})
        if "where" in q_lower or "哪里" in question:
            docs.append({"source": "question_type", "text": "This is a 'where' question requiring location understanding"})
        if "why" in q_lower or "为什么" in question:
            docs.append({"source": "question_type", "text": "This is a 'why' question requiring causal reasoning"})
        if "how" in q_lower or "如何" in question:
            docs.append({"source": "question_type", "text": "This is a 'how' question requiring process understanding"})

        return docs


_external_kb: ExternalKnowledgeBase | None = None


def get_external_kb() -> ExternalKnowledgeBase:
    global _external_kb
    if _external_kb is None:
        _external_kb = ExternalKnowledgeBase()
    return _external_kb


def build_enhanced_corpus(
    sample: dict[str, Any],
    *,
    retrieval_docs: list[dict[str, str]] | None = None,
) -> list[tuple[str, str]]:
    corpus: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    docs = retrieval_docs if retrieval_docs is not None else get_retrieval_docs(sample)

    for d in docs:
        if isinstance(d, dict) and d.get("text"):
            item = (str(d.get("source", "unknown")), str(d.get("text", "")))
            if item not in seen:
                seen.add(item)
                corpus.append(item)

    question = str(sample.get("question", ""))
    choices = _to_choice_list(sample.get("choices"))
    external_docs = get_external_kb().retrieve(question, choices)

    for d in external_docs:
        item = (str(d["source"]), str(d["text"]))
        if item not in seen:
            seen.add(item)
            corpus.append(item)

    return corpus


def _serialize_tool_logs(logs: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for log in logs:
        try:
            out.append(
                {
                    "tool_name": getattr(log, "tool_name", "unknown"),
                    "input_data": _jsonable(getattr(log, "input_data", {})),
                    "output_data": _jsonable(getattr(log, "output_data", "")),
                }
            )
        except Exception:
            out.append({"tool_name": "unknown", "input_data": {}, "output_data": str(log)})
    return out


def _serialize_evidence(evidence: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for e in evidence:
        out.append(
            {
                "source": getattr(e, "source", "unknown"),
                "claim": getattr(e, "claim", ""),
                "excerpt": getattr(e, "excerpt", ""),
                "confidence": float(getattr(e, "confidence", 0.0)),
                "supports": _jsonable(getattr(e, "supports", [])),
                "contradicts": _jsonable(getattr(e, "contradicts", [])),
                "evidence_type": getattr(e, "evidence_type", "text"),
                "premise_dependencies": _jsonable(getattr(e, "premise_dependencies", [])),
                "ambiguity": float(getattr(e, "ambiguity", 0.0)),
                "exclusivity": float(getattr(e, "exclusivity", 0.0)),
            }
        )
    return out


def _extract_kg_trace_from_logs(serialized_logs: list[dict[str, Any]]) -> dict[str, Any]:
    kg_calls = []
    lexical_calls = []
    for log in serialized_logs:
        tool_name = str(log.get("tool_name", ""))
        if tool_name == "retrieve_graph_evidence":
            kg_calls.append(log)
        elif tool_name == "retrieve_lexical_graph":
            lexical_calls.append(log)
    return {
        "retrieve_graph_evidence_calls": kg_calls,
        "retrieve_lexical_graph_calls": lexical_calls,
    }

_TEXT_CHOICE_DATASETS = {"aokvqa", "m3cot"}


def _extract_explicit_choice_from_reason(reason: str, choices: list[str]) -> str:
    """
    只在 reason 明确出现正确答案应为/option X should be selected 这类修正语句时，
    才允许用 reason 覆盖 decision.answer。
    """
    text = str(reason or "").strip()
    if not text or not choices:
        return ""

    patterns = [
        r"correct answer should be\s*(?:option\s*)?\(?([A-Z])\)?",
        r"correct answer is\s*(?:option\s*)?\(?([A-Z])\)?",
        r"corresponds to option\s*\(?([A-Z])\)?",
        r"option\s*\(?([A-Z])\)?\s*should be selected",
        r"option\s*\(?([A-Z])\)?\s*is the right option",
        r"应选\s*([A-Z])",
        r"正确答案(?:应)?为\s*([A-Z])",
        r"对应选项\s*([A-Z])",
    ]

    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if not m:
            continue
        label = m.group(1).upper()
        idx = ord(label) - ord("A")
        if 0 <= idx < len(choices):
            return str(choices[idx]).strip()

    return ""


def _normalize_agent_executor_answer(
    llm: Any,
    question: str,
    dataset_type: str,
    choices: list[str] | None,
    decision_answer: str,
    decision_reason: str,
    draft: str,
) -> str:
    """
    对 agent executor 的最终答案做后处理：
    先看 decision_reason 里是否出现了明确修正答案；
    再把 reason + answer + draft 合在一起做标准化抽取；
    仅对需要输出选项文本的数据集（AOKVQA / M3CoT）启用这条增强逻辑。
    """
    raw_answer = str(decision_answer or "").strip()
    raw_reason = str(decision_reason or "").strip()
    raw_draft = str(draft or "").strip()
    ds = str(dataset_type or "").lower()
    choices = choices or []

    if not choices:
        source_text = raw_reason or raw_answer or raw_draft
        normalized = extract_final_answer(
            llm=llm,
            question=question,
            draft=source_text,
            dataset_type=dataset_type,
            choices=choices,
        ).strip()
        return normalized or raw_answer

    # 只对需要输出选项文本的数据集做增强，避免影响 MMMU/CMMQA/ScienceQA 的字母输出
    if ds in _TEXT_CHOICE_DATASETS:
        explicit = _extract_explicit_choice_from_reason(raw_reason, choices)
        if explicit:
            return explicit

        direct_idx = _match_choice_text(raw_answer, choices)
        direct_choice = str(choices[direct_idx]).strip() if direct_idx is not None else ""

        source_text = "\n\n".join([x for x in [raw_reason, raw_answer, raw_draft] if x])
        normalized = extract_final_answer(
            llm=llm,
            question=question,
            draft=source_text or raw_answer or raw_draft,
            dataset_type=dataset_type,
            choices=choices,
        ).strip()

        norm_idx = _match_choice_text(normalized, choices)
        if norm_idx is not None:
            return str(choices[norm_idx]).strip()

        return direct_choice or normalized or raw_answer

    # 其余字母选项数据集保持原逻辑
    normalized = extract_final_answer(
        llm=llm,
        question=question,
        draft=raw_answer or raw_draft,
        dataset_type=dataset_type,
        choices=choices,
    ).strip()
    return normalized or raw_answer

def extract_final_answer(
    llm: Any,
    question: str,
    draft: str,
    dataset_type: str,
    choices: list[str] | None = None,
) -> str:
    choices = choices or []

    if llm is None:
        return _map_answer_to_choices(draft.strip(), choices) if choices else draft.strip()

    if choices:
        opts = []
        for i, ch in enumerate(choices[:10]):
            opts.append(f"{chr(ord('A') + i)}. {ch}")
        prompt = (
            "请从给定草稿中抽取最终标准答案。"
            "如果是选择题，只输出一个选项字母，不要解释。\n"
            f"题目: {question}\n"
            f"选项:\n{chr(10).join(opts)}\n"
            f"草稿:\n{draft[:3000]}"
        )
    else:
        prompt = (
            "请从给定草稿中抽取最终简洁答案，只输出答案本身，不要解释。\n"
            f"题目: {question}\n"
            f"草稿:\n{draft[:3000]}"
        )

    try:
        resp = llm.chat(
            [
                {"role": "system", "content": "假设你是专业的评测答案抽取助手。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        resp = (resp or "").strip()
        return _map_answer_to_choices(resp, choices) if choices else resp
    except Exception:
        fallback = draft.strip()
        return _map_answer_to_choices(fallback, choices) if choices else fallback


def run_direct_executor(
    llm: Any,
    sample: dict[str, Any],
    image_ref: str | None,
    baseline_context: str,
) -> ExecutorOutput:
    q = str(sample.get("question", ""))
    dataset_type = str(sample.get("dataset_type", "generic"))
    choices = _to_choice_list(sample.get("choices"))

    def _run_once(local_image_ref: str | None) -> str:
        return answer_with_context(
            llm=llm,
            question=q,
            context=baseline_context,
            image_ref=local_image_ref,
            dataset_type=dataset_type,
            choices=choices,
        )

    # 1) 先按原始设置执行
    try:
        pred = _run_once(image_ref)
        return ExecutorOutput(
            mode="direct_inference",
            answer=pred,
            context=baseline_context,
            retrieval_trace={
                "executor": "direct",
                "used_image": bool(image_ref),
            },
            final_confidence=0.0,
            decision_status="finalized" if str(pred or "").strip() else "fallback",
            decision_reason="direct_success" if str(pred or "").strip() else "direct_empty_answer",
        )
    except Exception as first_exc:
        first_error = f"{type(first_exc).__name__}: {first_exc}"

    # 2) 若是带图 baseline 失败，则自动退回 text-only baseline 再试一次
    if image_ref:
        try:
            pred = _run_once(None)
            return ExecutorOutput(
                mode="direct_inference",
                answer=pred,
                context=baseline_context,
                retrieval_trace={
                    "executor": "direct",
                    "used_image": False,
                    "image_fallback": "text_only_after_image_error",
                    "first_error": first_error,
                },
                error=None,
                final_confidence=0.0,
                decision_status="finalized" if str(pred or "").strip() else "fallback",
                decision_reason="direct_text_only_after_image_error",
            )
        except Exception as second_exc:
            second_error = f"{type(second_exc).__name__}: {second_exc}"
            return ExecutorOutput(
                mode="direct_inference",
                answer="",
                context=baseline_context,
                retrieval_trace={
                    "executor": "direct",
                    "used_image": False,
                    "image_fallback": "text_only_after_image_error",
                    "first_error": first_error,
                },
                error=f"{first_error} | text_only_retry_failed: {second_error}",
                final_confidence=0.0,
                decision_status="fallback",
                decision_reason="direct_image_and_text_failed",
            )

    # 3) 无图直接失败
    return ExecutorOutput(
        mode="direct_inference",
        answer="",
        context=baseline_context,
        retrieval_trace={"executor": "direct", "used_image": False},
        error=first_error,
        final_confidence=0.0,
        decision_status="fallback",
        decision_reason="direct_executor_exception",
    )


def run_lexical_executor(
    llm: Any,
    sample: dict[str, Any],
    image_ref: str | None,
    corpus: list[tuple[str, str]],
    route: str,
) -> ExecutorOutput:
    q = str(sample.get("question", ""))
    try:
        graph_hits, graph_context, route_meta = _build_route_aware_graph_context(
            route=route,
            question=q,
            corpus=corpus,
        )
        pred = answer_with_context(
            llm=llm,
            question=q,
            context=graph_context if graph_context else "",
            image_ref=image_ref,
            dataset_type=str(sample.get("dataset_type", "generic")),
            choices=_to_choice_list(sample.get("choices")),
        )
        return ExecutorOutput(
            mode="lexical_rag",
            answer=pred,
            context=graph_context,
            evidence=[
                {
                    "source": getattr(h, "source", "unknown"),
                    "claim": "",
                    "excerpt": getattr(h, "text", ""),
                    "confidence": float(getattr(h, "score", 0.0)),
                }
                for h in graph_hits
            ],
            retrieval_trace={
                "executor": "lexical",
                "route_meta": route_meta,
                "top_hits": [
                    {
                        "chunk_id": getattr(h, "chunk_id", ""),
                        "source": getattr(h, "source", ""),
                        "score": float(getattr(h, "score", 0.0)),
                        "reasons": getattr(h, "reasons", []),
                    }
                    for h in graph_hits
                ],
            },
            raw_hits=graph_hits,
        )
    except Exception as exc:
        return ExecutorOutput(
            mode="lexical_rag",
            answer="",
            retrieval_trace={"executor": "lexical"},
            error=str(exc),
        )

def _corpus_to_seed_text(corpus: list[tuple[str, str]], max_items: int = 4, max_chars: int = 1600) -> str:
    parts: list[str] = []
    for source, text in corpus[:max_items]:
        if not str(text).strip():
            continue
        parts.append(f"[{source}] {str(text)[:400]}")
    return "\n".join(parts)[:max_chars]

def _filter_agent_seed_corpus(corpus: list[tuple[str, str]]) -> list[tuple[str, str]]:
    blocked = {"choice_hint", "question_type", "common_sense"}
    out: list[tuple[str, str]] = []
    for source, text in corpus:
        if str(source).strip().lower() in blocked:
            continue
        if str(text).strip():
            out.append((source, text))
    return out

def run_vision_recheck_executor(
    llm: Any,
    sample: dict[str, Any],
    image_ref: str | None,
) -> ExecutorOutput:
    q = str(sample.get("question", ""))
    if llm is None:
        return ExecutorOutput(
            mode="vision_recheck",
            answer="",
            retrieval_trace={"executor": "vision_recheck", "mock": True},
            error="mock_mode_no_llm",
            decision_status="fallback",
        )

    if not image_ref:
        return ExecutorOutput(
            mode="vision_recheck",
            answer="",
            retrieval_trace={"executor": "vision_recheck", "reason": "no_image"},
            error="no_image_for_vision_recheck",
            decision_status="fallback",
        )

    logs: list[ToolObservation] = []
    evs: list[Evidence] = []
    parsed: dict[str, Any] = {}
    followups: list[str] = []

    try:
        search_tool = SearchTool()
        webpage_tool = WebpageTool()
        vision_tool = VisionTool(llm)
        lexical_graph_tool = LexicalGraphTool()
        byokg_provider = BYOKGRAGProvider()

        tool_router = ToolRouter(
            search_tool=search_tool,
            webpage_tool=webpage_tool,
            vision_tool=vision_tool,
            lexical_graph_tool=lexical_graph_tool,
            byokg_provider=byokg_provider,
            current_image_ref=image_ref,
        )
        tool_router.set_current_image_ref(image_ref)

        analyze_obs = tool_router.run(
            tool_name="analyze_image",
            args={"image_ref": image_ref, "question": q, "fallback_text": ""},
        )
        logs.append(analyze_obs)

        try:
            parsed = json.loads(str(analyze_obs.output_data or ""))
        except Exception:
            parsed = {}

        evs.extend(_structured_visual_to_seed_evidence(parsed, source="vision_recheck"))

        followups = _structured_visual_suggested_followups(parsed)
        for hint in followups[:2]:
            tool_router.set_current_image_ref(image_ref)

            region_obs = tool_router.run(
                tool_name="analyze_image_region",
                args={
                    "image_ref": image_ref,
                    "question": q,
                    "region_hint": hint,
                    "fallback_text": "",
                },
            )
            logs.append(region_obs)
            try:
                parsed_region = json.loads(str(region_obs.output_data or ""))
            except Exception:
                parsed_region = {}
            evs.extend(_structured_visual_to_seed_evidence(parsed_region, source=f"vision_recheck:{hint}"))

            if "clock" in hint or "text" in hint or "sign" in hint or "screen" in hint:
                tool_router.set_current_image_ref(image_ref)

                ocr_obs = tool_router.run(
                    tool_name="ocr_image_region",
                    args={"image_ref": image_ref, "region_hint": hint, "fallback_text": ""},
                )
                logs.append(ocr_obs)
                try:
                    ocr_parsed = json.loads(str(ocr_obs.output_data or ""))
                except Exception:
                    ocr_parsed = {}
                ocr_lines = ocr_parsed.get("ocr_lines", []) if isinstance(ocr_parsed, dict) else []
                if isinstance(ocr_lines, list) and ocr_lines:
                    lines: list[str] = []
                    for item in ocr_lines[:10]:
                        if not isinstance(item, dict):
                            continue
                        t = str(item.get("text", "")).strip()
                        region = str(item.get("region", "unknown")).strip()
                        conf = item.get("confidence", 0.8)
                        if t:
                            lines.append(f"[{region}] {t} (confidence={conf})")
                    if lines:
                        evs.append(
                            Evidence(
                                source=f"vision_recheck:{hint} #ocr",
                                claim="局部 OCR 事实",
                                excerpt="\n".join(lines),
                                confidence=0.84,
                                evidence_type="ocr",
                                premise_dependencies=[],
                                ambiguity=0.06,
                                exclusivity=0.42,
                            )
                        )

        serialized_logs = _serialize_tool_logs(logs)
        serialized_evidence = _serialize_evidence(evs)

        confidence = _fallback_decision_confidence_from_evidence(serialized_evidence)
        needs_escalation = any(
            e.get("evidence_type") == "ambiguity" or _safe_float(e.get("ambiguity", 0.0), 0.0) > 0.60
            for e in serialized_evidence
        )

        return ExecutorOutput(
            mode="vision_recheck",
            answer="",
            context=_extract_structured_vision_summary(parsed),
            evidence=serialized_evidence,
            tool_logs=serialized_logs,
            retrieval_trace={
                "executor": "vision_recheck",
                "needs_escalation": bool(needs_escalation),
                "suggested_followups": followups,
            },
            kg_trace={},
            final_confidence=confidence,
            decision_status="fallback" if needs_escalation else "finalized",
        )
    except Exception as exc:
        serialized_logs = _serialize_tool_logs(logs)
        serialized_evidence = _serialize_evidence(evs)
        fallback_confidence = _fallback_decision_confidence_from_evidence(serialized_evidence)

        return ExecutorOutput(
            mode="vision_recheck",
            answer="",
            context=_extract_structured_vision_summary(parsed),
            evidence=serialized_evidence,
            tool_logs=serialized_logs,
            retrieval_trace={
                "executor": "vision_recheck",
                "suggested_followups": followups,
                "exception_stage": "vision_recheck_executor",
            },
            kg_trace={},
            error=str(exc),
            final_confidence=fallback_confidence,
            decision_status="fallback",
        )

def _light_rag_proxy_schema(question: str) -> dict[str, Any]:
    return {
        "summary": f"Light-RAG proxy schema for: {question}",
        "node_types": ["Entity", "Concept", "DocumentChunk", "ImageFact", "OCRText"],
        "relation_types": [
            "related_to",
            "mentioned_in",
            "supports",
            "contradicts",
            "co_mentions",
            "about",
        ],
    }

def _build_light_rag_context_from_proxy_result(
    lexical_context: str,
    proxy_result: dict[str, Any],
    max_rows: int = 5,
    max_paths: int = 8,
) -> str:
    parts: list[str] = []

    if lexical_context.strip():
        parts.append("词图召回上下文:\n" + lexical_context[:2400])

    if not isinstance(proxy_result, dict):
        return "\n\n".join(parts).strip()

    graph_evidence = proxy_result.get("graph_evidence", {})
    if not isinstance(graph_evidence, dict):
        graph_evidence = {}

    query_results = graph_evidence.get("query_results", [])
    if isinstance(query_results, list) and query_results:
        row_lines: list[str] = []
        for row in query_results[:max_rows]:
            if not isinstance(row, dict):
                continue
            source = str(
                row.get("source", "")
                or row.get("doc_name", "")
                or row.get("title", "")
                or "proxy_doc"
            ).strip()
            text = str(
                row.get("text", "")
                or row.get("content", "")
                or row.get("snippet", "")
            ).strip()
            score = _safe_float(
                row.get("rerank_score", row.get("score", 0.0)),
                0.0,
            )
            if text:
                row_lines.append(f"[{source}] score={score:.3f} | {text[:500]}")
        if row_lines:
            parts.append("代理图谱命中文档:\n" + "\n".join(row_lines))

    paths = graph_evidence.get("paths", [])
    if isinstance(paths, list) and paths:
        path_lines: list[str] = []
        for path in paths[:max_paths]:
            if not isinstance(path, dict):
                continue
            head = str(path.get("subject", "") or path.get("head", "") or "").strip()
            rel = str(path.get("relation", "") or path.get("rel", "") or "related_to").strip()
            tail = str(path.get("object", "") or path.get("tail", "") or "").strip()
            if head or tail:
                path_lines.append(f"{head} -[{rel}]-> {tail}")
        if path_lines:
            parts.append("代理图谱路径:\n" + "\n".join(path_lines))

    linking_artifacts = proxy_result.get("linking_artifacts", {})
    if isinstance(linking_artifacts, dict):
        backend = str(linking_artifacts.get("external_backend", "")).strip()
        entities = linking_artifacts.get("linked_entities", linking_artifacts.get("entities", []))

        backend_lines: list[str] = []
        if backend:
            backend_lines.append(f"backend={backend}")

        if isinstance(entities, list) and entities:
            entity_texts = [str(x).strip() for x in entities[:12] if str(x).strip()]
            if entity_texts:
                backend_lines.append("entities=" + ", ".join(entity_texts))

        if backend_lines:
            parts.append("代理图谱链接信息:\n" + "\n".join(backend_lines))

    return "\n\n".join([p for p in parts if p.strip()]).strip()[:5000]

def _corpus_to_proxy_documents(
    corpus: list[tuple[str, str]],
    max_items: int = 6,
    max_chars: int = 800,
) -> list[dict[str, str]]:
    docs: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for source, text in corpus:
        src = _canonical_retrieval_source(source)
        txt = str(text).strip()
        if not txt:
            continue
        if src in _SYNTHETIC_SOURCES:
            continue

        item = (src, txt[:max_chars])
        if item in seen:
            continue
        seen.add(item)
        docs.append({"source": src, "text": txt[:max_chars]})

        if len(docs) >= max_items:
            break

    return docs

def _extract_proxy_image_candidates(proxy_result: dict[str, Any]) -> list[dict[str, str]]:
    if not isinstance(proxy_result, dict):
        return []

    ge = proxy_result.get("graph_evidence", {})
    if not isinstance(ge, dict):
        return []

    rows: list[dict[str, Any]] = []
    for key in ["kb_chunks", "query_results"]:
        value = ge.get(key, [])
        if isinstance(value, list):
            rows.extend([x for x in value if isinstance(x, dict)])

    out: list[dict[str, str]] = []
    seen: set[str] = set()

    for row in rows:
        row_source = str(row.get("doc_name", "") or row.get("source", "") or "").strip().lower()
        if "dbpedia" in row_source:
            continue

        explicit = row.get("image_candidates", [])
        if isinstance(explicit, list):
            for u in explicit:
                url = str(u).strip()
                if not url or url in seen:
                    continue
                seen.add(url)
                out.append(
                    {
                        "image_url": url,
                        "fallback_text": str(row.get("text", ""))[:300],
                        "source": str(row.get("doc_name", "") or row.get("source", "") or "kb_image"),
                    }
                )

    return out[:2]


def _structured_visual_doc_from_obs(obs: ToolObservation) -> str:
    try:
        parsed = json.loads(str(obs.output_data or ""))
    except Exception:
        return ""

    if not isinstance(parsed, dict):
        return ""

    lines: list[str] = []

    visible_facts = parsed.get("visible_facts", {})
    if isinstance(visible_facts, dict) and visible_facts:
        lines.append("图像可见事实:")
        lines.extend([f"{k}: {v}" for k, v in list(visible_facts.items())[:10]])

    ocr_facts = parsed.get("ocr_facts", [])
    if isinstance(ocr_facts, list) and ocr_facts:
        lines.append("OCR事实:")
        for item in ocr_facts[:8]:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            region = str(item.get("region", "unknown")).strip()
            if text:
                lines.append(f"[{region}] {text}")

    return "\n".join(lines).strip()

    
def run_light_rag_executor(
    llm: Any,
    sample: dict[str, Any],
    image_ref: str | None,
    corpus: list[tuple[str, str]],
    route: str,
) -> ExecutorOutput:
    q = str(sample.get("question", ""))
    dataset_type = str(sample.get("dataset_type", "generic"))
    choices = _to_choice_list(sample.get("choices"))

    graph_hits: list[Any] = []
    graph_context = ""
    route_meta: dict[str, Any] = {}
    proxy_documents: list[dict[str, str]] = []
    kg_obs: ToolObservation | None = None
    proxy_result: dict[str, Any] = {}
    extra_visual_logs: list[ToolObservation] = []
    extra_visual_docs: list[str] = []

    try:
        graph_hits, graph_context, route_meta = _build_route_aware_graph_context(
            route="light_rag",
            question=q,
            corpus=corpus,
        )

        # 优先把 lexical 命中的文档送进 proxy graph；没有命中时再退回原始 corpus
        hit_corpus: list[tuple[str, str]] = [
            (str(getattr(h, "source", "unknown")), str(getattr(h, "text", "")))
            for h in graph_hits
            if str(getattr(h, "text", "")).strip()
        ]
        proxy_documents = _corpus_to_proxy_documents(hit_corpus or corpus)

        if not proxy_documents and not graph_context.strip():
            return ExecutorOutput(
                mode="light_rag",
                answer="",
                context="",
                evidence=[],
                tool_logs=[],
                retrieval_trace={
                    "executor": "light_rag_proxy",
                    "route_meta": route_meta,
                    "top_hits": [],
                    "proxy_backend": "",
                    "proxy_document_count": 0,
                    "skipped_reason": "empty_corpus_before_proxy_graph",
                },
                kg_trace={
                    "retrieve_graph_evidence_calls": [],
                    "retrieve_lexical_graph_calls": [],
                },
                raw_hits=[],
                final_confidence=0.0,
                decision_status="fallback",
                decision_reason="empty_corpus_before_proxy_graph",
            )

        if llm is None:
            return run_lexical_executor(
                llm=llm,
                sample=sample,
                image_ref=image_ref,
                corpus=corpus,
                route=route,
            )

        search_tool = SearchTool()
        webpage_tool = WebpageTool()
        vision_tool = VisionTool(llm)
        lexical_graph_tool = LexicalGraphTool()
        byokg_provider = BYOKGRAGProvider()

        tool_router = ToolRouter(
            search_tool=search_tool,
            webpage_tool=webpage_tool,
            vision_tool=vision_tool,
            lexical_graph_tool=lexical_graph_tool,
            byokg_provider=byokg_provider,
            current_image_ref=image_ref,
        )
        tool_router.set_current_image_ref(image_ref)

        kg_obs = tool_router.run(
            tool_name="retrieve_graph_evidence",
            args={
                "question": q,
                "schema": _light_rag_proxy_schema(q),
                "graph_context": graph_context,
                "documents": proxy_documents,
                "route_mode": "light_rag",
            },
        )

        try:
            proxy_result = json.loads(str(kg_obs.output_data or ""))
        except Exception:
            proxy_result = {}

        # 只有原题没有图像输入时，才允许 light_rag 自动消费 proxy KG 返回的外部图片。
        if not image_ref:
            for candidate in _extract_proxy_image_candidates(proxy_result):
                image_url = str(candidate.get("image_url", "")).strip()
                if not image_url:
                    continue

                try:
                    img_obs = tool_router.run(
                        tool_name="analyze_image",
                        args={
                            "image_url": image_url,
                            "question": q,
                            "fallback_text": str(candidate.get("fallback_text", "")),
                            "allow_external_image_ref": True,
                        },
                    )
                except Exception as exc:
                    img_obs = ToolObservation(
                        tool_name="analyze_image",
                        input_data={
                            "image_url": image_url,
                            "question": q,
                            "fallback_text": str(candidate.get("fallback_text", "")),
                            "allow_external_image_ref": True,
                        },
                        output_data=f"TOOL_ERROR: {exc}",
                    )

                extra_visual_logs.append(img_obs)

                visual_doc = _structured_visual_doc_from_obs(img_obs)
                if visual_doc:
                    extra_visual_docs.append(visual_doc)

        enhanced_context = _build_light_rag_context_from_proxy_result(
            lexical_context=graph_context,
            proxy_result=proxy_result,
        )
        if extra_visual_docs:
            enhanced_context = (
                enhanced_context
                + "\n\n补充视觉证据:\n"
                + "\n\n".join(extra_visual_docs[:2])
            )

        pred = answer_with_context(
            llm=llm,
            question=q,
            context=enhanced_context,
            image_ref=image_ref,
            dataset_type=dataset_type,
            choices=choices,
        )

        evidence: list[dict[str, Any]] = [
            {
                "source": getattr(h, "source", "unknown"),
                "claim": "",
                "excerpt": getattr(h, "text", ""),
                "confidence": float(getattr(h, "score", 0.0)),
                "evidence_type": "graph",
            }
            for h in graph_hits
        ]

        if isinstance(proxy_result, dict):
            ge = proxy_result.get("graph_evidence", {})
            if isinstance(ge, dict):
                for row in ge.get("query_results", [])[:8]:
                    if not isinstance(row, dict):
                        continue
                    evidence.append(
                        {
                            "source": str(row.get("source", "") or row.get("doc_name", "") or "proxy_doc"),
                            "claim": "代理图谱命中文档",
                            "excerpt": str(row.get("text", ""))[:1000],
                            "confidence": _safe_float(row.get("rerank_score", row.get("score", 0.0)), 0.0),
                            "evidence_type": "graph",
                        }
                    )

                for p in ge.get("paths", [])[:8]:
                    if not isinstance(p, dict):
                        continue
                    head = str(p.get("subject", "") or p.get("head", "") or "").strip()
                    rel = str(p.get("relation", "") or p.get("rel", "") or "").strip()
                    tail = str(p.get("object", "") or p.get("tail", "") or "").strip()
                    path_text = f"{head} -[{rel}]-> {tail}".strip()
                    if path_text:
                        evidence.append(
                            {
                                "source": str(p.get("source", "proxy_graph")),
                                "claim": "代理图谱路径",
                                "excerpt": path_text,
                                "confidence": 0.65,
                                "evidence_type": "graph",
                            }
                        )

        serialized_tool_logs = []
        if kg_obs is not None:
            serialized_tool_logs.append(
                {
                    "tool_name": "retrieve_graph_evidence",
                    "input_data": _jsonable(kg_obs.input_data),
                    "output_data": _jsonable(kg_obs.output_data),
                }
            )

        for obs in extra_visual_logs:
            serialized_tool_logs.append(
                {
                    "tool_name": getattr(obs, "tool_name", "analyze_image"),
                    "input_data": _jsonable(getattr(obs, "input_data", {})),
                    "output_data": _jsonable(getattr(obs, "output_data", "")),
                }
            )

        return ExecutorOutput(
            mode="light_rag",
            answer=pred,
            context=enhanced_context,
            evidence=evidence,
            tool_logs=serialized_tool_logs,
            retrieval_trace={
                "executor": "light_rag_proxy",
                "route_meta": route_meta,
                "top_hits": [
                    {
                        "chunk_id": getattr(h, "chunk_id", ""),
                        "source": getattr(h, "source", ""),
                        "score": float(getattr(h, "score", 0.0)),
                        "reasons": getattr(h, "reasons", []),
                    }
                    for h in graph_hits
                ],
                "proxy_backend": proxy_result.get("linking_artifacts", {}).get("external_backend", "")
                if isinstance(proxy_result, dict) else "",
                "proxy_document_count": len(proxy_documents),
            },
            kg_trace={
                "retrieve_graph_evidence_calls": serialized_tool_logs,
                "retrieve_lexical_graph_calls": [],
            },
            raw_hits=graph_hits,
        )

    except Exception as exc:
        partial_evidence: list[dict[str, Any]] = [
            {
                "source": getattr(h, "source", "unknown"),
                "claim": "",
                "excerpt": getattr(h, "text", ""),
                "confidence": float(getattr(h, "score", 0.0)),
                "evidence_type": "graph",
            }
            for h in graph_hits
        ]

        if isinstance(proxy_result, dict):
            ge = proxy_result.get("graph_evidence", {})
            if isinstance(ge, dict):
                for row in ge.get("query_results", [])[:8]:
                    if not isinstance(row, dict):
                        continue
                    partial_evidence.append(
                        {
                            "source": str(row.get("source", "") or row.get("doc_name", "") or "proxy_doc"),
                            "claim": "代理图谱命中文档",
                            "excerpt": str(row.get("text", ""))[:1000],
                            "confidence": _safe_float(row.get("rerank_score", row.get("score", 0.0)), 0.0),
                            "evidence_type": "graph",
                        }
                    )

        serialized_tool_logs = []
        if kg_obs is not None:
            serialized_tool_logs.append(
                {
                    "tool_name": "retrieve_graph_evidence",
                    "input_data": _jsonable(kg_obs.input_data),
                    "output_data": _jsonable(kg_obs.output_data),
                }
            )

        for obs in extra_visual_logs:
            serialized_tool_logs.append(
                {
                    "tool_name": getattr(obs, "tool_name", "analyze_image"),
                    "input_data": _jsonable(getattr(obs, "input_data", {})),
                    "output_data": _jsonable(getattr(obs, "output_data", "")),
                }
            )

        fallback_context = _build_light_rag_context_from_proxy_result(
            lexical_context=graph_context,
            proxy_result=proxy_result,
        )
        if extra_visual_docs:
            fallback_context = (
                fallback_context
                + "\n\n补充视觉证据:\n"
                + "\n\n".join(extra_visual_docs[:2])
            )

        fallback_confidence = _fallback_decision_confidence_from_evidence(partial_evidence)

        return ExecutorOutput(
            mode="light_rag",
            answer="",
            context=fallback_context,
            evidence=partial_evidence,
            tool_logs=serialized_tool_logs,
            retrieval_trace={
                "executor": "light_rag_proxy",
                "route": route,
                "route_meta": route_meta,
                "proxy_document_count": len(proxy_documents),
                "exception_stage": "light_rag_executor",
            },
            kg_trace={
                "retrieve_graph_evidence_calls": serialized_tool_logs,
                "retrieve_lexical_graph_calls": [],
            },
            raw_hits=graph_hits,
            error=str(exc),
            final_confidence=fallback_confidence,
            decision_status="fallback",
            decision_reason="light_rag_executor_exception",
        )
        
        
def _pairwise_option_distinguishable(
    llm: Any,
    *,
    question: str,
    choices: list[str],
    chosen_answer: str,
    draft: str,
    serialized_evidence: list[dict[str, Any]],
) -> tuple[bool, str]:
    if llm is None or not choices or not str(chosen_answer or "").strip():
        return True, ""

    chosen = _resolve_choice_text_for_pairwise(chosen_answer, choices)
    chosen_norm = _normalize_cmp_text(chosen)
    distractors = [str(c).strip() for c in choices if _normalize_cmp_text(str(c)) != chosen_norm]
    if not distractors:
        return True, ""

    distractor = distractors[0]
    draft_norm = _normalize_cmp_text(draft)
    for cand in distractors:
        if _normalize_cmp_text(cand) in draft_norm:
            distractor = cand
            break

    evidence_preview: list[str] = []
    for e in serialized_evidence[:6]:
        evidence_preview.append(
            f"- [{e.get('source', 'unknown')}] "
            f"{str(e.get('claim', '')).strip()} | "
            f"{str(e.get('excerpt', '')).strip()[:180]}"
        )

    data = llm.chat_json(
        [
            {
                "role": "system",
                "content": (
                    "判断当前证据是否足以把 chosen option 与 distractor option 明确区分开。"
                    '只输出 JSON：{"distinguishable": true/false, "reason": "..."}'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n"
                    f"Chosen option: {chosen}\n"
                    f"Distractor option: {distractor}\n"
                    f"Draft:\n{draft[:1200]}\n\n"
                    f"Evidence:\n" + "\n".join(evidence_preview)
                ),
            },
        ],
        fallback={"distinguishable": True, "reason": "pairwise_check_fallback"},
        route="reasoning",
    )

    if not isinstance(data, dict):
        return True, ""

    ok = bool(data.get("distinguishable", True))
    reason = str(data.get("reason", "")).strip()
    return ok, reason


def run_agent_executor(
    llm: Any,
    sample: dict[str, Any],
    image_ref: str | None,
    corpus: list[tuple[str, str]],
    route: str,
) -> ExecutorOutput:
    if llm is None:
        return ExecutorOutput(
            mode="agentic_rag",
            answer="",
            retrieval_trace={"executor": "agent", "route": route, "mock": True},
            error="mock_mode_no_llm",
            decision_status="fallback",
            decision_reason="mock_mode_no_llm",
        )

    q = str(sample.get("question", ""))
    dataset_type = str(sample.get("dataset_type", "generic"))
    choices = _to_choice_list(sample.get("choices"))

    sub_questions: list[SubQuestion] = []
    research_main_question = q
    all_evidence: list[Evidence] = []
    all_logs: list[ToolObservation] = []
    verified_evidence: list[Evidence] = []
    critique = None
    visual_hint = ""
    seed_context = ""
    seed_corpus: list[tuple[str, str]] = []
    unresolved_conflict = False

    try:
        search_tool = SearchTool()
        webpage_tool = WebpageTool()
        vision_tool = VisionTool(llm)
        lexical_graph_tool = LexicalGraphTool()
        byokg_provider = BYOKGRAGProvider()

        tool_router = ToolRouter(
            search_tool=search_tool,
            webpage_tool=webpage_tool,
            vision_tool=vision_tool,
            lexical_graph_tool=lexical_graph_tool,
            byokg_provider=byokg_provider,
            current_image_ref=image_ref,
        )

        planner = PlannerAgent(llm)
        researcher = ResearcherAgent(llm, tool_router)
        critic = CriticAgent(llm)
        verifier = VerifierAgent(llm)
        synthesizer = SynthesizerAgent(llm)

        tool_router.set_current_image_ref(image_ref)

        visual_struct: dict[str, Any] = {}

        seed_corpus = _filter_agent_seed_corpus(corpus)
        seed_context = _corpus_to_seed_text(seed_corpus)

        seed_logs: list[ToolObservation] = []
        seed_evidence: list[Evidence] = []
        seed_docs: list[dict[str, str]] = [
            {"source": str(source), "text": str(text)[:800]}
            for source, text in seed_corpus[:6]
            if str(text).strip()
        ]

        if image_ref:
            try:
                tool_router.set_current_image_ref(image_ref)
                seed_obs = tool_router.run(
                    tool_name="analyze_image",
                    args={
                        "image_ref": image_ref,
                        "question": q,
                        "fallback_text": "",
                    },
                )

                try:
                    visual_struct = json.loads(str(seed_obs.output_data or ""))
                except Exception:
                    visual_struct = {}

                if not isinstance(visual_struct, dict):
                    visual_struct = {}

                if visual_struct:
                    visual_hint = _extract_structured_vision_summary(visual_struct)
                    seed_logs.append(
                        ToolObservation(
                            tool_name="seed_local_image_analysis",
                            input_data={
                                "question": q,
                                "source": "resolved_local_image",
                                "image_ref": image_ref,
                                "analysis_mode": "structured_json",
                            },
                            output_data=json.dumps(visual_struct, ensure_ascii=False),
                        )
                    )
                    seed_evidence.extend(
                        _structured_visual_to_seed_evidence(
                            visual_struct,
                            source="local_image",
                        )
                    )
                    seed_docs.extend(
                        _structured_visual_to_seed_docs(
                            visual_struct,
                            source="local_image_file",
                        )
                    )
            except Exception:
                visual_struct = {}
                visual_hint = ""

        if seed_context:
            seed_logs.append(
                ToolObservation(
                    tool_name="seed_local_documents",
                    input_data={"count": min(len(seed_corpus), 4)},
                    output_data=seed_context,
                )
            )
            for source, text in seed_corpus[:3]:
                if str(text).strip():
                    seed_evidence.append(
                        Evidence(
                            source=str(source),
                            claim="本地文档候选证据",
                            excerpt=str(text)[:500],
                            confidence=0.65,
                            evidence_type="webpage_fact" if str(source).startswith("http") else "text",
                            premise_dependencies=[],
                            ambiguity=0.15,
                            exclusivity=0.25,
                        )
                    )

        planning_question = q
        if choices:
            planning_question += "\n选项:\n" + "\n".join(
                [f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(choices[:10])]
            )
        if visual_hint:
            planning_question += (
                "\n图像分析摘要:\n"
                f"{visual_hint}\n"
                "注意：上面的图像内容已区分可见事实、OCR事实和歧义项。"
                "后续优先围绕这些事实验证，不要把候选解释当成已证实事实。"
            )
        if seed_context:
            planning_question += f"\n已有本地证据:\n{seed_context}"

        sub_questions = planner.plan(planning_question)
        research_main_question = planning_question

        all_evidence = list(seed_evidence)
        all_logs = list(seed_logs)

        for sq in sub_questions:
            current_sq = sq
            used_qs = [sq.question]
            max_rewrites = max(
                1,
                int(getattr(settings, "max_subquestion_rewrites", 2) or 2),
            )

            for _ in range(max_rewrites):
                tool_router.set_current_image_ref(image_ref)
                evs, logs = researcher.investigate(
                    main_question=research_main_question,
                    sub_question=current_sq,
                    seed_docs=seed_docs,
                    route_mode=route,
                )
                all_evidence.extend(evs)
                all_logs.extend(logs)

                grounding_accept = True
                grounding_reason = ""

                for log in reversed(logs):
                    if log.tool_name == "subquestion_grounding_check":
                        try:
                            parsed = json.loads(str(log.output_data or ""))
                        except Exception:
                            parsed = {}
                        grounding_accept = bool(parsed.get("accept", True))
                        grounding_reason = str(parsed.get("reason", "")).strip()
                        break

                if grounding_accept:
                    break

                current_sq = researcher._rewrite_subquestion_from_visual_feedback(
                    main_question=research_main_question,
                    bad_sub_question=current_sq,
                    logs=all_logs,
                    failure_reason=grounding_reason or "not_grounded_to_image",
                    used_questions=used_qs,
                )
                used_qs.append(current_sq.question)

        verified_evidence = verifier.verify(q, all_evidence)
        critique = critic.critique(q, verified_evidence)

        if not critique.sufficient:
            for fq in critique.follow_up_questions[:2]:
                pseudo_sq = SubQuestion(question=fq, intent="follow_up", priority=99)
                tool_router.set_current_image_ref(image_ref)
                evs, logs = researcher.investigate(
                    main_question=research_main_question,
                    sub_question=pseudo_sq,
                    seed_docs=seed_docs,
                    route_mode=route,
                )
                all_evidence.extend(evs)
                all_logs.extend(logs)

            if critique.must_reinspect_image and image_ref:
                vision_recheck_out = run_vision_recheck_executor(
                    llm=llm,
                    sample=sample,
                    image_ref=image_ref,
                )
                all_logs.extend(
                    [
                        ToolObservation(
                            tool_name=x.get("tool_name", "vision_recheck"),
                            input_data=x.get("input_data", {}),
                            output_data=str(x.get("output_data", "")),
                        )
                        for x in vision_recheck_out.tool_logs
                    ]
                )
                for e in vision_recheck_out.evidence:
                    all_evidence.append(
                        Evidence(
                            source=str(e.get("source", "vision_recheck")),
                            claim=str(e.get("claim", "")),
                            excerpt=str(e.get("excerpt", "")),
                            confidence=_safe_float(e.get("confidence", 0.0), 0.0),
                            supports=[str(x) for x in e.get("supports", [])]
                            if isinstance(e.get("supports", []), list) else [],
                            contradicts=[str(x) for x in e.get("contradicts", [])]
                            if isinstance(e.get("contradicts", []), list) else [],
                            evidence_type=str(e.get("evidence_type", "text")),
                            premise_dependencies=[str(x) for x in e.get("premise_dependencies", [])]
                            if isinstance(e.get("premise_dependencies", []), list) else [],
                            ambiguity=_safe_float(e.get("ambiguity", 0.0), 0.0),
                            exclusivity=_safe_float(e.get("exclusivity", 0.0), 0.0),
                        )
                    )

            verified_evidence = verifier.verify(q, all_evidence)
            critique = critic.critique(q, verified_evidence)

        state = ResearchState(
            question=research_main_question,
            sub_questions=sub_questions,
            evidence=verified_evidence,
            tool_logs=all_logs,
        )

        draft = synthesizer.synthesize(state)
        decision = synthesizer.decide(state)

        serialized_logs = _serialize_tool_logs(all_logs)
        serialized_evidence = _serialize_evidence(verified_evidence)

        critique_confidence = _safe_float(getattr(critique, "final_confidence", 0.0), 0.0)
        decision_confidence = _safe_float(
            getattr(decision, "confidence", critique_confidence),
            critique_confidence,
        )
        if decision_confidence <= 0.0:
            decision_confidence = critique_confidence
        if decision_confidence <= 0.0:
            decision_confidence = _fallback_decision_confidence_from_evidence(serialized_evidence)

        decision_answer = str(getattr(decision, "answer", "") or "").strip()
        decision_abstain = bool(getattr(decision, "abstain", False))
        decision_reason = str(getattr(decision, "reason", "") or "").strip()

        unresolved_conflict = (
            bool(getattr(critique, "has_unresolved_contradiction", False))
            or _detect_unresolved_conflict(serialized_evidence)
        )

        normalized_answer = decision_answer
        if choices:
            normalized_answer = _normalize_agent_executor_answer(
                llm=llm,
                question=q,
                dataset_type=dataset_type,
                choices=choices,
                decision_answer=decision_answer,
                decision_reason=decision_reason,
                draft=draft,
            ).strip() or decision_answer

        should_finalize, finalize_reason = _should_finalize_agent_answer(
            critique=critique,
            serialized_evidence=serialized_evidence,
            decision_confidence=decision_confidence,
            normalized_answer=normalized_answer,
            unresolved_conflict=unresolved_conflict,
        )

        if should_finalize and len(choices) >= 2:
            pairwise_ok, pairwise_reason = _pairwise_option_distinguishable(
                llm=llm,
                question=q,
                choices=choices,
                chosen_answer=normalized_answer,
                draft=draft,
                serialized_evidence=serialized_evidence,
            )
            if not pairwise_ok:
                should_finalize = False
                finalize_reason = (
                    f"pairwise_not_distinguishable: {pairwise_reason}"
                    if pairwise_reason else "pairwise_not_distinguishable"
                )

        if should_finalize:
            decision_abstain = False
            decision_answer = normalized_answer.strip()
            decision_status = "finalized"
            if not decision_reason:
                decision_reason = finalize_reason
        else:
            decision_abstain = True
            decision_answer = ""
            decision_status = "abstained"
            decision_reason = finalize_reason

        decision_status = "abstained" if decision_abstain else "finalized"

        return ExecutorOutput(
            mode="agentic_rag",
            answer=decision_answer,
            context=draft,
            evidence=serialized_evidence,
            tool_logs=serialized_logs,
            retrieval_trace={
                "executor": "agent",
                "route": route,
                "sub_questions": [
                    {
                        "question": getattr(sq, "question", ""),
                        "intent": getattr(sq, "intent", ""),
                        "priority": getattr(sq, "priority", 0),
                    }
                    for sq in sub_questions
                ],
                "critique": {
                    "sufficient": critique.sufficient,
                    "follow_up_questions": critique.follow_up_questions,
                    "missing_dimensions": critique.missing_dimensions,
                    "must_reinspect_image": getattr(critique, "must_reinspect_image", False),
                    "must_abort_finalization": getattr(critique, "must_abort_finalization", False),
                    "dominant_failure_mode": getattr(critique, "dominant_failure_mode", ""),
                    "final_confidence": critique_confidence,
                    "has_unresolved_contradiction": bool(getattr(critique, "has_unresolved_contradiction", False)),
                },
                "decision": {
                    "answer": decision_answer,
                    "confidence": decision_confidence,
                    "abstain": decision_abstain,
                    "reason": decision_reason,
                    "has_unresolved_conflict": unresolved_conflict,
                },
                "has_seed_local_image_analysis": bool(visual_hint),
                "has_seed_local_documents": bool(seed_context),
                "seed_document_count": min(len(seed_corpus), 4),
            },
            kg_trace=_extract_kg_trace_from_logs(serialized_logs),
            final_confidence=decision_confidence,
            decision_status=decision_status,
            decision_reason=decision_reason,
        )
    except Exception as exc:
        effective_evidence = verified_evidence if verified_evidence else all_evidence
        serialized_logs = _serialize_tool_logs(all_logs)
        serialized_evidence = _serialize_evidence(effective_evidence)
        fallback_confidence = _fallback_decision_confidence_from_evidence(serialized_evidence)

        critique_trace = {}
        if critique is not None:
            critique_trace = {
                "sufficient": getattr(critique, "sufficient", False),
                "follow_up_questions": getattr(critique, "follow_up_questions", []),
                "missing_dimensions": getattr(critique, "missing_dimensions", []),
                "must_reinspect_image": getattr(critique, "must_reinspect_image", False),
                "must_abort_finalization": getattr(critique, "must_abort_finalization", False),
                "dominant_failure_mode": getattr(critique, "dominant_failure_mode", ""),
                "final_confidence": _safe_float(getattr(critique, "final_confidence", 0.0), 0.0),
            }

        return ExecutorOutput(
            mode="agentic_rag",
            answer="",
            context="",
            evidence=serialized_evidence,
            tool_logs=serialized_logs,
            retrieval_trace={
                "executor": "agent",
                "route": route,
                "sub_questions": [
                    {
                        "question": getattr(sq, "question", ""),
                        "intent": getattr(sq, "intent", ""),
                        "priority": getattr(sq, "priority", 0),
                    }
                    for sq in sub_questions
                ],
                "critique": critique_trace,
                "has_seed_local_image_analysis": bool(visual_hint),
                "has_seed_local_documents": bool(seed_context),
                "seed_document_count": min(len(seed_corpus), 4) if seed_corpus else 0,
                "exception_stage": "agent_executor",
            },
            kg_trace=_extract_kg_trace_from_logs(serialized_logs),
            error=str(exc),
            final_confidence=fallback_confidence,
            decision_status="fallback",
            decision_reason="agent_executor_exception",
        )

_legacy_run_agent_executor = run_agent_executor


def _workflow_critique_to_dict(critique: Any) -> dict[str, Any]:
    if critique is None:
        return {}
    return {
        "sufficient": bool(getattr(critique, "sufficient", False)),
        "follow_up_questions": list(getattr(critique, "follow_up_questions", []) or []),
        "missing_dimensions": list(getattr(critique, "missing_dimensions", []) or []),
        "must_reinspect_image": bool(getattr(critique, "must_reinspect_image", False)),
        "must_abort_finalization": bool(getattr(critique, "must_abort_finalization", False)),
        "dominant_failure_mode": str(getattr(critique, "dominant_failure_mode", "")),
        "final_confidence": _safe_float(getattr(critique, "final_confidence", 0.0), 0.0),
        "has_unresolved_contradiction": bool(getattr(critique, "has_unresolved_contradiction", False)),
    }


def run_agent_executor(
    llm: Any,
    sample: dict[str, Any],
    image_ref: str | None,
    corpus: list[tuple[str, str]],
    route: str,
) -> ExecutorOutput:
    if llm is None:
        return ExecutorOutput(
            mode="agentic_rag",
            answer="",
            retrieval_trace={"executor": "agent", "route": route, "mock": True},
            error="mock_mode_no_llm",
            decision_status="fallback",
            decision_reason="mock_mode_no_llm",
        )

    q = str(sample.get("question", ""))
    dataset_type = str(sample.get("dataset_type", "generic"))
    choices = _to_choice_list(sample.get("choices"))

    try:
        search_tool = SearchTool()
        webpage_tool = WebpageTool()
        vision_tool = VisionTool(llm)
        lexical_graph_tool = LexicalGraphTool()
        byokg_provider = BYOKGRAGProvider()

        tool_router = ToolRouter(
            search_tool=search_tool,
            webpage_tool=webpage_tool,
            vision_tool=vision_tool,
            lexical_graph_tool=lexical_graph_tool,
            byokg_provider=byokg_provider,
            current_image_ref=image_ref,
        )

        planner = PlannerAgent(llm)
        researcher = ResearcherAgent(llm, tool_router)
        critic = CriticAgent(llm)
        verifier = VerifierAgent(llm)
        synthesizer = SynthesizerAgent(llm)

        seed_corpus = _filter_agent_seed_corpus(corpus)
        seed_docs = [
            {"source": str(source), "text": str(text)[:800]}
            for source, text in seed_corpus[:6]
            if str(text).strip()
        ]

        workflow_question = q
        if choices:
            workflow_question += "\n閫夐」:\n" + "\n".join(
                [f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(choices[:10])]
            )

        workflow = DeepResearchWorkflow(
            planner=planner,
            researcher=researcher,
            critic=critic,
            verifier=verifier,
            synthesizer=synthesizer,
        )
        workflow_result = workflow.run_with_trace(
            question=workflow_question,
            image_ref=image_ref,
            route_mode=route,
            seed_docs=seed_docs,
        )

        state = workflow_result.state
        serialized_logs = _serialize_tool_logs(state.tool_logs)
        serialized_evidence = _serialize_evidence(state.evidence)
        critique = workflow_result.final_critique
        decision = workflow_result.decision

        critique_confidence = _safe_float(getattr(critique, "final_confidence", 0.0), 0.0)
        decision_confidence = _safe_float(
            getattr(decision, "confidence", critique_confidence),
            critique_confidence,
        )
        if decision_confidence <= 0.0:
            decision_confidence = _fallback_decision_confidence_from_evidence(serialized_evidence)

        decision_answer = str(getattr(decision, "answer", "") or "").strip()
        decision_reason = str(getattr(decision, "reason", "") or "").strip()
        decision_abstain = bool(getattr(decision, "abstain", False))

        if choices and decision_answer:
            decision_answer = _normalize_agent_executor_answer(
                llm=llm,
                question=q,
                dataset_type=dataset_type,
                choices=choices,
                decision_answer=decision_answer,
                decision_reason=decision_reason,
                draft=workflow_result.report,
            ).strip() or decision_answer

        unresolved_conflict = (
            bool(getattr(critique, "has_unresolved_contradiction", False))
            or _detect_unresolved_conflict(serialized_evidence)
        )
        should_finalize, finalize_reason = _should_finalize_agent_answer(
            critique=critique,
            serialized_evidence=serialized_evidence,
            decision_confidence=decision_confidence,
            normalized_answer=decision_answer,
            unresolved_conflict=unresolved_conflict,
        )

        if should_finalize and len(choices) >= 2:
            pairwise_ok, pairwise_reason = _pairwise_option_distinguishable(
                llm=llm,
                question=q,
                choices=choices,
                chosen_answer=decision_answer,
                draft=workflow_result.report,
                serialized_evidence=serialized_evidence,
            )
            if not pairwise_ok:
                should_finalize = False
                finalize_reason = (
                    f"pairwise_not_distinguishable: {pairwise_reason}"
                    if pairwise_reason else "pairwise_not_distinguishable"
                )

        if workflow_result.decision_status == "fallback":
            decision_status = "fallback"
            decision_answer = ""
            decision_reason = decision_reason or "workflow_fallback"
            decision_abstain = True
        elif should_finalize:
            decision_status = "finalized"
            decision_abstain = False
            decision_reason = decision_reason or finalize_reason
        else:
            decision_status = "abstained"
            decision_answer = ""
            decision_abstain = True
            decision_reason = finalize_reason

        return ExecutorOutput(
            mode="agentic_rag",
            answer=decision_answer,
            context=workflow_result.report,
            evidence=serialized_evidence,
            tool_logs=serialized_logs,
            retrieval_trace={
                "executor": "agent",
                "executor_impl": "workflow.run_with_trace",
                "route": route,
                "sub_questions": [
                    {
                        "question": getattr(sq, "question", ""),
                        "intent": getattr(sq, "intent", ""),
                        "priority": getattr(sq, "priority", 0),
                    }
                    for sq in getattr(state, "sub_questions", [])
                ],
                "critique": _workflow_critique_to_dict(critique),
                "critique_trace": workflow_result.critique_trace,
                "route_trace": workflow_result.route_trace,
                "error_trace": workflow_result.error_trace,
                "decision": {
                    "answer": decision_answer,
                    "confidence": decision_confidence,
                    "abstain": decision_abstain,
                    "reason": decision_reason,
                    "has_unresolved_conflict": unresolved_conflict,
                },
                "seed_document_count": min(len(seed_corpus), 4),
            },
            kg_trace=_extract_kg_trace_from_logs(serialized_logs),
            final_confidence=decision_confidence,
            decision_status=decision_status,
            decision_reason=decision_reason,
        )
    except Exception as exc:
        return ExecutorOutput(
            mode="agentic_rag",
            answer="",
            context="",
            evidence=[],
            tool_logs=[],
            retrieval_trace={
                "executor": "agent",
                "executor_impl": "workflow.run_with_trace",
                "route": route,
                "exception_stage": "agent_executor_workflow_bridge",
            },
            kg_trace={},
            error=str(exc),
            final_confidence=0.0,
            decision_status="fallback",
            decision_reason="agent_executor_workflow_bridge_exception",
        )


def run_route_executor(
    llm: Any,
    sample: dict[str, Any],
    image_ref: str | None,
    corpus: list[tuple[str, str]],
    route: str,
) -> ExecutorOutput:

    # 保持同一个图像引用向下游透传，避免中途改写/丢失
    resolved_image_ref = str(image_ref).strip() if image_ref else None
    if resolved_image_ref == "":
        resolved_image_ref = None

    # 统一 route 名称，避免外部传入大小写/空白差异
    effective_route = str(route or "").strip().lower() or "direct_inference"

    # direct 分支仍然只使用当前传入的 baseline_context；
    baseline_context = "\n\n".join([f"[{s}] {t[:600]}" for s, t in corpus[:4]])

    if effective_route == "direct_inference":
        return run_direct_executor(
            llm=llm,
            sample=sample,
            image_ref=resolved_image_ref,
            baseline_context=baseline_context,
        )

    # vision_recheck 只能在有图时运行；没图就降级
    if effective_route == "vision_recheck":
        if not resolved_image_ref:
            fallback_route = "light_rag" if corpus else "direct_inference"
            if fallback_route == "direct_inference":
                return run_direct_executor(
                    llm=llm,
                    sample=sample,
                    image_ref=None,
                    baseline_context=baseline_context,
                )
            if llm is None:
                return run_lexical_executor(
                    llm=llm,
                    sample=sample,
                    image_ref=None,
                    corpus=corpus,
                    route="light_rag",
                )
            return run_light_rag_executor(
                llm=llm,
                sample=sample,
                image_ref=None,
                corpus=corpus,
                route="light_rag",
            )

        vision_out = run_vision_recheck_executor(
            llm=llm,
            sample=sample,
            image_ref=resolved_image_ref,
        )

        # 视觉复检已经足够，就直接返回；
        # 若仍需升级，再继续向 light/full 分支传同一个 image_ref
        if not vision_out.retrieval_trace.get("needs_escalation"):
            return vision_out

        if corpus:
            # 有可检索语料时，优先 light_rag
            if llm is None:
                return run_lexical_executor(
                    llm=llm,
                    sample=sample,
                    image_ref=resolved_image_ref,
                    corpus=corpus,
                    route="light_rag",
                )
            return run_light_rag_executor(
                llm=llm,
                sample=sample,
                image_ref=resolved_image_ref,
                corpus=corpus,
                route="light_rag",
            )

        # 没有语料时，vision_recheck 已经是最后一次可靠视觉补救
        return vision_out

    # light_rag：没有可检索语料时不允许硬跑，优先回退到 vision_recheck / direct
    if effective_route == "light_rag":
        if not corpus:
            if resolved_image_ref:
                return run_vision_recheck_executor(
                    llm=llm,
                    sample=sample,
                    image_ref=resolved_image_ref,
                )
            return run_direct_executor(
                llm=llm,
                sample=sample,
                image_ref=None,
                baseline_context=baseline_context,
            )

        if llm is None:
            return run_lexical_executor(
                llm=llm,
                sample=sample,
                image_ref=resolved_image_ref,
                corpus=corpus,
                route="light_rag",
            )

        return run_light_rag_executor(
            llm=llm,
            sample=sample,
            image_ref=resolved_image_ref,
            corpus=corpus,
            route="light_rag",
        )

    # full_rag：mock / 无 LLM 时降级到 lexical；
    # 没有语料时，不硬跑 full_rag，改走 vision_recheck / direct
    if effective_route == "full_rag":
        if not corpus:
            if resolved_image_ref:
                return run_vision_recheck_executor(
                    llm=llm,
                    sample=sample,
                    image_ref=resolved_image_ref,
                )
            return run_direct_executor(
                llm=llm,
                sample=sample,
                image_ref=None,
                baseline_context=baseline_context,
            )

        if llm is None:
            return run_lexical_executor(
                llm=llm,
                sample=sample,
                image_ref=resolved_image_ref,
                corpus=corpus,
                route="light_rag",
            )

        return run_agent_executor(
            llm=llm,
            sample=sample,
            image_ref=resolved_image_ref,
            corpus=corpus,
            route="full_rag",
        )

    # 未知 route 的兜底：尽量保守，不要把图像引用丢掉
    if corpus:
        if llm is None:
            return run_lexical_executor(
                llm=llm,
                sample=sample,
                image_ref=resolved_image_ref,
                corpus=corpus,
                route="light_rag",
            )
        return run_light_rag_executor(
            llm=llm,
            sample=sample,
            image_ref=resolved_image_ref,
            corpus=corpus,
            route="light_rag",
        )

    if resolved_image_ref:
        return run_vision_recheck_executor(
            llm=llm,
            sample=sample,
            image_ref=resolved_image_ref,
        )

    return run_direct_executor(
        llm=llm,
        sample=sample,
        image_ref=None,
        baseline_context=baseline_context,
    )


def _looks_like_visual_label_mcq(
    question: str,
    choices: list[str] | None,
    image_ref: str | None,
    dataset_type: str = "",
) -> bool:
    if not image_ref:
        return False

    choice_list = [str(x).strip() for x in (choices or []) if str(x).strip()]
    if len(choice_list) < 2:
        return False

    if str(dataset_type or "").strip().lower() not in _MC_DATASETS:
        return False

    q = str(question or "").strip().lower()

    ocr_like = [
        "read the text", "what does the sign say", "logo", "label", "poster",
        "screen", "clock", "timer", "ocr",
        "文字", "标志", "标签", "海报", "屏幕", "时钟", "计时器",
    ]
    if any(k in q for k in ocr_like):
        return False

    q_cues = [
        "what type", "which type", "what kind", "which kind",
        "what category", "which category",
        "哪一类", "属于哪类", "属于哪一类", "类型", "类别",
    ]
    if not any(k in q for k in q_cues):
        return False

    norm_choices = [c.lower() for c in choice_list]
    if any(len(c.split()) > 3 for c in norm_choices):
        return False

    categoryish = {
        "domestic", "wild", "aquatic", "stuffed",
        "urban", "rural", "residential", "commercial", "private", "public",
        "indoor", "outdoor", "day", "night", "daytime", "nighttime",
        "mammal", "bird", "reptile", "amphibian", "fish", "insect",
        "宠物", "野生", "水生", "标本", "城市", "乡村", "室内", "室外",
        "白天", "夜晚", "哺乳动物", "鸟类", "爬行动物", "两栖动物", "鱼类", "昆虫",
    }
    category_hits = sum(1 for c in norm_choices if c in categoryish)

    return category_hits >= max(2, len(norm_choices) // 2)


def _resolve_choice_text_for_pairwise(answer: str, choices: list[str]) -> str:
    raw = str(answer or "").strip()
    if not raw or not choices:
        return raw

    idx = _extract_choice_label_from_text(raw, len(choices))
    if idx is not None:
        return str(choices[idx]).strip()

    idx = _match_choice_text(raw, choices)
    if idx is not None:
        return str(choices[idx]).strip()

    return raw



def determine_route_dynamically(
    difficulty_score: float,
    question: str,
    image_ref: str | None,
    has_documents: bool,
    question_type: str = "",
    dataset_type: str = "",
    baseline_pred: str = "",
    baseline_exact_match: bool = False,
    choices: list[str] | None = None,
) -> str:
    global _route_thresholds
    q33 = _route_thresholds.get("q33", 0.34)
    q67 = _route_thresholds.get("q67", 0.67)
    dataset_type = str(dataset_type or "").lower()

    if difficulty_score > q67:
        base_route = "full_rag"
    elif difficulty_score > q33:
        base_route = "light_rag"
    else:
        base_route = "direct_inference"

    if baseline_exact_match:
        return "direct_inference"

    q_lower = str(question or "").lower()
    baseline_uncertain = not str(baseline_pred or "").strip()

    if _looks_like_visual_label_mcq(
        question=question,
        choices=choices,
        image_ref=image_ref,
        dataset_type=dataset_type,
    ):
        return "vision_recheck"

    knowledge_intensive_keywords = [
        "who", "what", "where", "why", "when",
        "谁", "什么", "哪里", "为什么", "什么时候",
        "history", "science", "geography", "名人",
    ]
    is_knowledge_intensive = any(kw in q_lower for kw in knowledge_intensive_keywords)

    visual_ambiguity_keywords = [
        "which part of the day", "what time", "how many", "what color",
        "read the text", "what does the sign say", "clock", "timer",
        "logo", "sign", "label", "poster", "screen", "shape",
        "时段", "几点", "白天还是晚上", "颜色", "数量", "文字", "标志", "海报", "时钟", "计时器",
        "标识", "标签", "屏幕", "形状",
    ]
    is_visual_ambiguity = any(kw in q_lower for kw in visual_ambiguity_keywords)

    if image_ref and is_visual_ambiguity:
        return "vision_recheck"

    if image_ref and base_route == "direct_inference" and not has_documents:
        return "vision_recheck"

    if is_knowledge_intensive and base_route == "direct_inference":
        if baseline_uncertain or has_documents:
            return "light_rag"

    return base_route

_MC_DATASETS = {"aokvqa", "m3cot", "mmmu_pro", "cmmqa", "scienceqa"}

_BLOCKED_RETRIEVAL_SOURCES = {
    "scienceqa": {"solution"},
    "mmmu_pro": {"solution"},
}

_SYNTHETIC_SOURCES = {"choice_hint", "question_type", "common_sense"}


def _canonical_retrieval_source(src: str) -> str:
    s = str(src or "").strip().lower()
    mapping = {
        "rationales": "rationale",
        "explain": "explanation",
        "reasoning_chain": "reasoning_chains",
        "reasoningchains": "reasoning_chains",
    }
    return mapping.get(s, s or "unknown")



def _append_unique_retrieval_doc(
    out: list[dict[str, str]],
    seen: set[tuple[str, str]],
    *,
    src: str,
    txt: str,
    blocked_sources: set[str],
    allow_synthetic: bool,
) -> None:
    src = _canonical_retrieval_source(src)
    txt = str(txt or "").strip()
    if not txt:
        return

    # rationale / explain / explanation 这类标注在 retrieval 语料中允许保留
    if src not in _RETRIEVAL_ANNOTATION_SOURCES:
        if src in blocked_sources:
            return
        if (not allow_synthetic) and src in _SYNTHETIC_SOURCES:
            return

    key = (src, txt)
    if key in seen:
        return
    seen.add(key)
    out.append({"source": src, "text": txt})

def _dataset_allows_annotation_retrieval(dataset_type: str) -> bool:
    enabled = bool(getattr(settings, "include_annotation_retrieval_docs", False))
    if not enabled:
        return False

    allow_datasets = [
        str(x).strip().lower()
        for x in getattr(settings, "annotation_retrieval_datasets", []) or []
        if str(x).strip()
    ]
    if not allow_datasets:
        return True

    return str(dataset_type or "").strip().lower() in allow_datasets


def _doc_allowed_by_policy(
    source: str,
    *,
    dataset_type: str,
    use_case: str,   # "baseline" / "retrieval"
    allow_synthetic: bool = False,
    include_annotation_docs: bool = False,
) -> bool:
    src = _canonical_retrieval_source(source)

    blocked_sources = set(_BLOCKED_RETRIEVAL_SOURCES.get(dataset_type, set()))
    if src in blocked_sources:
        return False

    if use_case == "baseline":
        if src in _RETRIEVAL_ANNOTATION_SOURCES:
            return False
        if src in _SYNTHETIC_SOURCES:
            return False
        return True

    if use_case == "retrieval":
        if src in _SYNTHETIC_SOURCES and not allow_synthetic:
            return False
        if src in _RETRIEVAL_ANNOTATION_SOURCES and not include_annotation_docs:
            return False
        return True

    return True


def _synthesize_annotation_docs_from_original(sample: dict[str, Any]) -> list[dict[str, str]]:
    """
    从 __original_record 再补一遍 annotation docs：
    - rationale / rationales
    - explain / explanation
    - reasoning_chains
    """
    original = sample.get("__original_record", {})
    if not isinstance(original, dict):
        return []

    out: list[dict[str, str]] = []

    for field_name, source_name in [
        ("rationale", "rationale"),
        ("explain", "explanation"),
        ("explanation", "explanation"),
    ]:
        value = original.get(field_name)
        if isinstance(value, str) and value.strip():
            out.append({"source": source_name, "text": value.strip()})

    rationales = _to_text_list(original.get("rationales"))
    if rationales:
        out.append({"source": "rationale", "text": "\n".join(rationales)})

    reasoning_chains = original.get("reasoning_chains")
    if isinstance(reasoning_chains, list) and reasoning_chains:
        chain_lines: list[str] = []
        for idx, chain in enumerate(reasoning_chains, start=1):
            if isinstance(chain, (list, tuple)) and len(chain) >= 3:
                chain_lines.append(f"{idx}. {chain[0]} --{chain[1]}--> {chain[2]}")
            elif isinstance(chain, str) and chain.strip():
                chain_lines.append(f"{idx}. {chain.strip()}")
        if chain_lines:
            out.append({"source": "reasoning_chains", "text": "\n".join(chain_lines)})
    elif isinstance(reasoning_chains, str) and reasoning_chains.strip():
        out.append({"source": "reasoning_chains", "text": reasoning_chains.strip()})

    return out


def _dataset_allows_annotation_retrieval(dataset_type: str) -> bool:
    enabled = bool(getattr(settings, "include_annotation_retrieval_docs", False))
    if not enabled:
        return False

    allow_datasets = [
        str(x).strip().lower()
        for x in getattr(settings, "annotation_retrieval_datasets", []) or []
        if str(x).strip()
    ]
    if not allow_datasets:
        return True

    return str(dataset_type or "").strip().lower() in allow_datasets


def get_retrieval_docs(
    sample: dict[str, Any],
    *,
    allow_synthetic: bool = False,
    include_annotation_docs: bool | None = None,
) -> list[dict[str, str]]:
    dataset_type = str(sample.get("dataset_type", "")).lower()
    docs = sample.get("documents", [])

    if include_annotation_docs is None:
        include_annotation_docs = _dataset_allows_annotation_retrieval(dataset_type)

    out: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for d in docs:
        if not isinstance(d, dict):
            continue
        src = d.get("source", "unknown")
        txt = str(d.get("text", "")).strip()
        if not txt:
            continue
        if not _doc_allowed_by_policy(
            src,
            dataset_type=dataset_type,
            use_case="retrieval",
            allow_synthetic=allow_synthetic,
            include_annotation_docs=bool(include_annotation_docs),
        ):
            continue
        _append_unique_retrieval_doc(
            out,
            seen,
            src=src,
            txt=txt,
            blocked_sources=set(),
            allow_synthetic=True,
        )

    if include_annotation_docs:
        for d in _synthesize_annotation_docs_from_original(sample):
            src = d.get("source", "unknown")
            txt = str(d.get("text", "")).strip()
            if not txt:
                continue
            if not _doc_allowed_by_policy(
                src,
                dataset_type=dataset_type,
                use_case="retrieval",
                allow_synthetic=allow_synthetic,
                include_annotation_docs=True,
            ):
                continue
            _append_unique_retrieval_doc(
                out,
                seen,
                src=src,
                txt=txt,
                blocked_sources=set(),
                allow_synthetic=True,
            )

    return out



def get_baseline_docs(sample: dict[str, Any]) -> list[dict[str, str]]:
    docs = sample.get("documents", [])
    dataset_type = str(sample.get("dataset_type", "")).lower()

    out: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for d in docs:
        if not isinstance(d, dict):
            continue

        src = d.get("source", "unknown")
        txt = str(d.get("text", "")).strip()
        if not txt:
            continue
        if not _doc_allowed_by_policy(
            src,
            dataset_type=dataset_type,
            use_case="baseline",
            allow_synthetic=False,
            include_annotation_docs=False,
        ):
            continue

        src_norm = _canonical_retrieval_source(src)
        key = (src_norm, txt)
        if key in seen:
            continue
        seen.add(key)
        out.append({"source": src_norm, "text": txt})

    return out

def _select_eval_sample(data: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    """
    按比例做确定性随机抽样：
    - sample_ratio >= 1 或 <=0 时保持全量；
    - 抽样后按原始下标排序，避免破坏后续 shard 的稳定性；
    - shard 在抽样之后再执行。
    """
    try:
        ratio = float(getattr(args, "sample_ratio", 1.0) or 1.0)
    except Exception:
        ratio = 1.0

    if ratio <= 0.0 or ratio >= 1.0 or len(data) <= 1:
        return data

    try:
        seed = int(getattr(args, "sample_seed", 42) or 42)
    except Exception:
        seed = 42

    n = max(1, int(round(len(data) * ratio)))
    n = min(n, len(data))

    rng = random.Random(seed)
    selected_indices = sorted(rng.sample(range(len(data)), n))

    return [data[i] for i in selected_indices]

def _select_eval_shard(data: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    """
    按样本下标做确定性分片：
    - num_shards=1 时保持原逻辑不变；
    - num_shards>1 时，每个进程只处理 idx % num_shards == shard_id 的样本。
    """
    try:
        num_shards = int(getattr(args, "num_shards", 1) or 1)
    except Exception:
        num_shards = 1

    try:
        shard_id = int(getattr(args, "shard_id", 0) or 0)
    except Exception:
        shard_id = 0

    num_shards = max(1, num_shards)

    if not (0 <= shard_id < num_shards):
        raise ValueError(f"Invalid shard_id={shard_id}; expected 0 <= shard_id < num_shards={num_shards}")

    if num_shards <= 1:
        return data

    return [sample for idx, sample in enumerate(data) if idx % num_shards == shard_id]


def run_eval(dataset_path: Path, args: argparse.Namespace) -> dict:
    all_data = load_dataset(dataset_path, args)
    sampled_data = _select_eval_sample(all_data, args)

    ablation = _resolve_ablation(str(getattr(args, "ablation_profile", "all_on")))

    effective_web_search_tool_enabled = _resolve_effective_web_search_tool_enabled(
        args=args,
        ablation=ablation,
    )
    settings.enable_web_search_tool = effective_web_search_tool_enabled

    if not effective_web_search_tool_enabled:
        settings.enable_visual_semantic_web_search = False

    vllm_local_media_root = _get_vllm_local_media_root(args)
    if vllm_local_media_root is not None:
        vllm_local_media_root.mkdir(parents=True, exist_ok=True)

    # 注意：动态路由阈值在分片前估计，避免不同 shard 的 route 阈值不一致。
    if sampled_data:
        route_values = [estimate_question_difficulty(sample).overall for sample in sampled_data]
        q33_route, q67_route = _tertile_thresholds(route_values)
        set_route_thresholds(q33_route, q67_route)

    # 真正执行时才按 shard 过滤。
    data = _select_eval_shard(sampled_data, args)

    llm = None
    if not args.mock:
        from models import QwenVLClient

        llm = QwenVLClient()
        required_runtime_methods = [
            "_normalize_image_ref_for_vllm",
            "chat_with_image",
            "chat_json",
            "image_ref_to_data_url",
        ]
        missing_runtime_methods = [m for m in required_runtime_methods if not hasattr(llm, m)]
        if missing_runtime_methods:
            raise RuntimeError(
                "Runtime QwenVLClient is not the patched version. "
                f"Missing methods: {missing_runtime_methods}"
            )

    records: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    skipped_stats: dict[str, int] = defaultdict(int)

    resume_path = args.resume_from
    if resume_path is not None:
        resumed = _load_resume_records(resume_path)
        for r in resumed:
            if not isinstance(r, dict):
                continue
            k = r.get("sample_key")
            if isinstance(k, str) and k:
                seen_keys.add(k)
                records.append(r)

    progress = tqdm(data, desc="Evaluating", unit="sample", dynamic_ncols=True)

    for sample in progress:
        sample_key = _record_key(sample)
        if sample_key in seen_keys:
            continue

        q = str(sample["question"])
        gold = str(sample["answer"]).strip()
        dataset_type = str(sample.get("dataset_type", "")).lower()
        choices = _to_choice_list(sample.get("choices"))

        if not gold:
            skipped_stats[f"{dataset_type or 'generic'}_empty_gold"] += 1
            continue

        def _apply_choice_map(answer: str) -> str:
            raw = str(answer or "").strip()
            if not raw or not choices:
                return raw

            mapper = globals().get("_map_answer_to_choices")
            if callable(mapper):
                try:
                    return str(mapper(raw, choices))
                except Exception:
                    pass

            # 没有全局 helper 时，做一个最小 choice 映射
            try:
                decoded_text, _, decoded_idx = _decode_choice_answer(raw, choices)
                if decoded_idx is not None and 0 <= decoded_idx < len(choices):
                    return str(choices[decoded_idx])
                if decoded_text and decoded_text in choices:
                    return decoded_text
            except Exception:
                pass

            raw_norm = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", raw.lower())
            best_choice = raw
            best_score = -1.0
            for ch in choices:
                ch_text = str(ch)
                ch_norm = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", ch_text.lower())
                if not ch_norm:
                    continue
                if raw_norm == ch_norm:
                    return ch_text
                if raw_norm and (raw_norm in ch_norm or ch_norm in raw_norm):
                    score = min(len(raw_norm), len(ch_norm)) / max(len(raw_norm), len(ch_norm), 1)
                    if score > best_score:
                        best_score = score
                        best_choice = ch_text
            return best_choice if best_score >= 0.45 else raw

        # difficulty
        if ablation.use_difficulty:
            diff = estimate_question_difficulty(sample)
            difficulty_overall = diff.overall
        else:
            diff = estimate_question_difficulty(
                {
                    "dataset_type": str(sample.get("dataset_type", "generic")),
                    "question": str(sample.get("question", "")),
                    "choices": sample.get("choices", []),
                }
            )
            difficulty_overall = diff.overall

        # image
        image_ref, image_source = resolve_image_reference(
            sample,
            image_root=args.image_root,
            vllm_local_media_root=vllm_local_media_root,
        )

        # docs
        baseline_docs = get_baseline_docs(sample)
        retrieval_docs = get_retrieval_docs(sample)

        retrieval_seed_trace: dict[str, Any] = {
            "enabled": bool(getattr(settings, "enable_gpt_retrieval_seed", False)),
            "called": False,
            "ok": False,
            "docs_added": 0,
            "model": str(getattr(settings, "gpt_retrieval_seed_model", "gpt-5.1") or "gpt-5.1"),
            "error": "",
        }

        retrieval_docs, retrieval_seed_trace = augment_retrieval_docs_with_gpt(
            llm=llm,
            sample=sample,
            image_ref=image_ref,
            retrieval_docs=retrieval_docs,
        )

        baseline_corpus = [
            (str(d.get("source", "unknown")), str(d.get("text", "")))
            for d in baseline_docs
            if d.get("text")
        ]

        enhanced_corpus = build_enhanced_corpus(sample, retrieval_docs=retrieval_docs)
        has_retrieval_corpus = bool(enhanced_corpus)

        # baseline run
        baseline_context = "\n\n".join([f"[{s}] {t[:600]}" for s, t in baseline_corpus[:4]])
        baseline_out = run_direct_executor(
            llm=llm,
            sample=sample,
            image_ref=image_ref,
            baseline_context=baseline_context,
        )

        baseline_pred = _apply_choice_map(str(baseline_out.answer or "").strip())
        baseline_exact = exact_match_for_eval(baseline_pred, gold, sample) if baseline_pred else 0
        route_out: ExecutorOutput | None = None
        planned_route = ""
        normal_route = ""
        
        # optional independent CLIP recall + BGE rerank + MLLM branch
        clip_rerank_mllm_enabled = bool(getattr(settings, "enable_clip_rerank_mllm", False))

        # optional independent GraphRAG + MLLM branch
        graphrag_mllm_enabled = bool(getattr(settings, "enable_graphrag_mllm", False))

        # optional independent mR2AG + MLLM branch
        mr2ag_mllm_enabled = bool(getattr(settings, "enable_mr2ag_mllm", False))

        # optional independent Self-RAG + MLLM branch
        self_rag_mllm_enabled = bool(getattr(settings, "enable_self_rag_mllm", False))

        if clip_rerank_mllm_enabled:
            planned_route = "clip_rerank_mllm"
            normal_route = "clip_rerank_mllm"

            try:
                from clip_rerank_mllm import run_clip_rerank_mllm

                # 优先使用外部检索候选 enhanced_corpus；若为空，则退回 baseline_corpus，
                # 但整个分支仍不调用 SearchTool/WebpageTool/agent search_web。
                clip_corpus = enhanced_corpus if enhanced_corpus else baseline_corpus

                clip_result = run_clip_rerank_mllm(
                    llm=llm,
                    sample=sample,
                    image_ref=image_ref,
                    corpus=clip_corpus,
                    answer_fn=answer_with_context,
                    choices=choices,
                    map_answer_fn=_apply_choice_map,
                )

                route_out = ExecutorOutput(
                    mode=str(clip_result.get("mode", "clip_rerank_mllm")),
                    answer=str(clip_result.get("answer", "") or ""),
                    context=str(clip_result.get("context", "") or ""),
                    evidence=clip_result.get("evidence", []) if isinstance(clip_result.get("evidence", []), list) else [],
                    tool_logs=[],
                    retrieval_trace=clip_result.get("retrieval_trace", {}) if isinstance(clip_result.get("retrieval_trace", {}), dict) else {},
                    kg_trace={},
                    raw_hits=[],
                    error=clip_result.get("error"),
                    final_confidence=_safe_float(clip_result.get("final_confidence", 0.0), 0.0),
                    decision_status=str(clip_result.get("decision_status", "finalized") or "finalized"),
                    decision_reason=str(clip_result.get("decision_reason", "clip_rerank_mllm") or "clip_rerank_mllm"),
                )
            except Exception as exc:
                route_out = ExecutorOutput(
                    mode="clip_rerank_mllm",
                    answer="",
                    context="",
                    evidence=[],
                    tool_logs=[],
                    retrieval_trace={
                        "executor": "clip_rerank_mllm",
                        "exception": f"{type(exc).__name__}: {exc}",
                    },
                    kg_trace={},
                    raw_hits=[],
                    error=f"{type(exc).__name__}: {exc}",
                    final_confidence=0.0,
                    decision_status="fallback",
                    decision_reason="clip_rerank_mllm_outer_exception",
                )
        
        
        elif graphrag_mllm_enabled:
            planned_route = "graphrag_mllm"
            normal_route = "graphrag_mllm"

            try:
                from graphrag_mllm import run_graphrag_mllm

                # GraphRAG + MLLM 只使用本地候选语料 / 已构建 GraphRAG workspace，
                # 不调用 SearchTool / WebpageTool / agent search_web。
                graphrag_corpus = enhanced_corpus if enhanced_corpus else baseline_corpus

                graphrag_result = run_graphrag_mllm(
                    llm=llm,
                    sample=sample,
                    image_ref=image_ref,
                    corpus=graphrag_corpus,
                    answer_fn=answer_with_context,
                    choices=choices,
                    map_answer_fn=_apply_choice_map,
                )

                route_out = ExecutorOutput(
                    mode=str(graphrag_result.get("mode", "graphrag_mllm")),
                    answer=str(graphrag_result.get("answer", "") or ""),
                    context=str(graphrag_result.get("context", "") or ""),
                    evidence=graphrag_result.get("evidence", []) if isinstance(graphrag_result.get("evidence", []), list) else [],
                    tool_logs=graphrag_result.get("tool_logs", []) if isinstance(graphrag_result.get("tool_logs", []), list) else [],
                    retrieval_trace=graphrag_result.get("retrieval_trace", {}) if isinstance(graphrag_result.get("retrieval_trace", {}), dict) else {},
                    kg_trace=graphrag_result.get("kg_trace", {}) if isinstance(graphrag_result.get("kg_trace", {}), dict) else {},
                    raw_hits=graphrag_result.get("raw_hits", []) if isinstance(graphrag_result.get("raw_hits", []), list) else [],
                    error=graphrag_result.get("error"),
                    final_confidence=_safe_float(graphrag_result.get("final_confidence", 0.0), 0.0),
                    decision_status=str(graphrag_result.get("decision_status", "finalized") or "finalized"),
                    decision_reason=str(graphrag_result.get("decision_reason", "graphrag_mllm") or "graphrag_mllm"),
                )
            except Exception as exc:
                route_out = ExecutorOutput(
                    mode="graphrag_mllm",
                    answer="",
                    context="",
                    evidence=[],
                    tool_logs=[],
                    retrieval_trace={
                        "executor": "graphrag_mllm",
                        "no_web_search": True,
                        "exception": f"{type(exc).__name__}: {exc}",
                    },
                    kg_trace={},
                    raw_hits=[],
                    error=f"{type(exc).__name__}: {exc}",
                    final_confidence=0.0,
                    decision_status="fallback",
                    decision_reason="graphrag_mllm_outer_exception",
                )
        
        
        elif mr2ag_mllm_enabled:
            planned_route = "mr2ag_mllm"
            normal_route = "mr2ag_mllm"

            try:
                from mr2ag_mllm import run_mr2ag_mllm

                # mR2AG + MLLM 只使用本地候选证据，不调用 SearchTool/WebpageTool。
                # 优先 enhanced_corpus；若为空则退回 baseline_corpus。
                mr2ag_corpus = enhanced_corpus if enhanced_corpus else baseline_corpus

                mr2ag_result = run_mr2ag_mllm(
                    llm=llm,
                    sample=sample,
                    image_ref=image_ref,
                    corpus=mr2ag_corpus,
                    answer_fn=answer_with_context,
                    choices=choices,
                    map_answer_fn=_apply_choice_map,
                )

                route_out = ExecutorOutput(
                    mode=str(mr2ag_result.get("mode", "mr2ag_mllm")),
                    answer=str(mr2ag_result.get("answer", "") or ""),
                    context=str(mr2ag_result.get("context", "") or ""),
                    evidence=mr2ag_result.get("evidence", []) if isinstance(mr2ag_result.get("evidence", []), list) else [],
                    tool_logs=mr2ag_result.get("tool_logs", []) if isinstance(mr2ag_result.get("tool_logs", []), list) else [],
                    retrieval_trace=mr2ag_result.get("retrieval_trace", {}) if isinstance(mr2ag_result.get("retrieval_trace", {}), dict) else {},
                    kg_trace=mr2ag_result.get("kg_trace", {}) if isinstance(mr2ag_result.get("kg_trace", {}), dict) else {},
                    raw_hits=mr2ag_result.get("raw_hits", []) if isinstance(mr2ag_result.get("raw_hits", []), list) else [],
                    error=mr2ag_result.get("error"),
                    final_confidence=_safe_float(mr2ag_result.get("final_confidence", 0.0), 0.0),
                    decision_status=str(mr2ag_result.get("decision_status", "finalized") or "finalized"),
                    decision_reason=str(mr2ag_result.get("decision_reason", "mr2ag_mllm") or "mr2ag_mllm"),
                )
            except Exception as exc:
                route_out = ExecutorOutput(
                    mode="mr2ag_mllm",
                    answer="",
                    context="",
                    evidence=[],
                    tool_logs=[],
                    retrieval_trace={
                        "executor": "mr2ag_mllm",
                        "no_web_search": True,
                        "exception": f"{type(exc).__name__}: {exc}",
                    },
                    kg_trace={},
                    raw_hits=[],
                    error=f"{type(exc).__name__}: {exc}",
                    final_confidence=0.0,
                    decision_status="fallback",
                    decision_reason="mr2ag_mllm_outer_exception",
                )

        elif self_rag_mllm_enabled:
            planned_route = "self_rag_mllm"
            normal_route = "self_rag_mllm"

            try:
                from self_rag_mllm import run_self_rag_mllm

                # Self-RAG + MLLM 只用本地候选证据，不调用 SearchTool/WebpageTool。
                # 优先 enhanced_corpus；若为空则退回 baseline_corpus。
                self_rag_corpus = enhanced_corpus if enhanced_corpus else baseline_corpus

                self_rag_result = run_self_rag_mllm(
                    llm=llm,
                    sample=sample,
                    image_ref=image_ref,
                    corpus=self_rag_corpus,
                    answer_fn=answer_with_context,
                    choices=choices,
                    map_answer_fn=_apply_choice_map,
                )

                route_out = ExecutorOutput(
                    mode=str(self_rag_result.get("mode", "self_rag_mllm")),
                    answer=str(self_rag_result.get("answer", "") or ""),
                    context=str(self_rag_result.get("context", "") or ""),
                    evidence=self_rag_result.get("evidence", []) if isinstance(self_rag_result.get("evidence", []), list) else [],
                    tool_logs=self_rag_result.get("tool_logs", []) if isinstance(self_rag_result.get("tool_logs", []), list) else [],
                    retrieval_trace=self_rag_result.get("retrieval_trace", {}) if isinstance(self_rag_result.get("retrieval_trace", {}), dict) else {},
                    kg_trace=self_rag_result.get("kg_trace", {}) if isinstance(self_rag_result.get("kg_trace", {}), dict) else {},
                    raw_hits=self_rag_result.get("raw_hits", []) if isinstance(self_rag_result.get("raw_hits", []), list) else [],
                    error=self_rag_result.get("error"),
                    final_confidence=_safe_float(self_rag_result.get("final_confidence", 0.0), 0.0),
                    decision_status=str(self_rag_result.get("decision_status", "finalized") or "finalized"),
                    decision_reason=str(self_rag_result.get("decision_reason", "self_rag_mllm") or "self_rag_mllm"),
                )
            except Exception as exc:
                route_out = ExecutorOutput(
                    mode="self_rag_mllm",
                    answer="",
                    context="",
                    evidence=[],
                    tool_logs=[],
                    retrieval_trace={
                        "executor": "self_rag_mllm",
                        "no_web_search": True,
                        "exception": f"{type(exc).__name__}: {exc}",
                    },
                    kg_trace={},
                    raw_hits=[],
                    error=f"{type(exc).__name__}: {exc}",
                    final_confidence=0.0,
                    decision_status="fallback",
                    decision_reason="self_rag_mllm_outer_exception",
                )

        else:
            # dynamic route
            normal_route = determine_route_dynamically(
                difficulty_score=difficulty_overall,
                question=q,
                image_ref=image_ref,
                has_documents=has_retrieval_corpus,
                question_type=str(sample.get("dataset_type", "")),
                dataset_type=dataset_type,
                baseline_pred=baseline_pred,
                baseline_exact_match=bool(baseline_exact),
                choices=choices,
            )

            # ablation 中关闭 KG-RAG 时，不能再走 light_rag / full_rag
            planned_route = normal_route if ablation.use_kg_rag else "direct_inference"

            # direct_inference 已经跑过 baseline_out，直接复用，避免重复推理
            if planned_route == "direct_inference":
                route_out = baseline_out
            else:
                route_out = run_route_executor(
                    llm=llm,
                    sample=sample,
                    image_ref=image_ref,
                    corpus=enhanced_corpus,
                    route=planned_route,
                )

        # final answer assembly
        if route_out is None:
            route_out = baseline_out
            planned_route = planned_route or "direct_inference"
            normal_route = normal_route or planned_route
        route_answer = _apply_choice_map(str(route_out.answer or "").strip())
        route_pred = route_answer

        route_confidence = _safe_float(getattr(route_out, "final_confidence", 0.0), 0.0)
        decision_status = str(getattr(route_out, "decision_status", "finalized") or "finalized")
        decision_reason = str(getattr(route_out, "decision_reason", "") or "")

        final_pred = route_pred
        final_confidence = route_confidence
        final_answer_source = "route"
        final_trace_source = "route"

        fallback_answer = ""
        fallback_confidence = 0.0
        
        route_error = str(getattr(route_out, "error", "") or "").strip()
        decision_reason_l = str(decision_reason or "").lower()

        should_fallback = (
            llm is not None
            and (
                not final_pred
                or decision_status == "fallback"
                or bool(route_error)
                or "exception" in decision_reason_l
                or (
                    bool(getattr(settings, "enable_fallback_on_abstain", False))
                    and decision_status == "abstained"
                )
            )
        )

        if should_fallback:
            fallback_docs = get_fallback_docs(sample, retrieval_docs=retrieval_docs)
            fallback_context = "\n\n".join(
                [
                    f"[{str(d.get('source', 'unknown'))}] {str(d.get('text', ''))[:600]}"
                    for d in fallback_docs[:6]
                    if d.get("text")
                ]
            )
            fallback_out = run_direct_executor(
                llm=llm,
                sample=sample,
                image_ref=image_ref,
                baseline_context=fallback_context,
            )

            mapped_fallback = _apply_choice_map(str(fallback_out.answer or "").strip())

            if mapped_fallback and not fallback_out.error:
                fallback_answer = mapped_fallback
                fallback_confidence = _safe_float(getattr(fallback_out, "final_confidence", 0.55), 0.55)

                final_pred = fallback_answer
                final_confidence = max(final_confidence, fallback_confidence)
                final_answer_source = "fallback_direct"
                final_trace_source = "fallback_direct"
                decision_status = "fallback"
                decision_reason = (
                    (decision_reason + "|fallback_direct")
                    if decision_reason else
                    "fallback_direct"
                )
            elif baseline_pred and not baseline_out.error:
                fallback_answer = baseline_pred
                fallback_confidence = _safe_float(getattr(baseline_out, "final_confidence", 0.55), 0.55)

                final_pred = fallback_answer
                final_confidence = max(final_confidence, fallback_confidence)
                final_answer_source = "baseline_fallback"
                final_trace_source = "baseline_fallback"
                decision_status = "fallback"
                decision_reason = (
                    (decision_reason + "|fallback_to_baseline")
                    if decision_reason else
                    "fallback_to_baseline"
                )

        # optional sft refine
        if (
            ablation.use_sft
            and final_pred
            and decision_status == "finalized"
            and route_out.mode != "baseline_short_circuit"
        ):
            try:
                sft_pred = refine_with_sft(
                    llm=llm,
                    question=q,
                    first_answer=final_pred,
                    context=route_out.context,
                    image_ref=image_ref,
                    dataset_type=str(sample.get("dataset_type", "generic")),
                    choices=choices,
                )
                sft_pred = _apply_choice_map(str(sft_pred or "").strip())
                if sft_pred and _normalize_cmp_text(sft_pred) != _normalize_cmp_text(final_pred):
                    final_pred = sft_pred
                    decision_reason = (decision_reason + "|sft_refined") if decision_reason else "sft_refined"
                    final_confidence = min(final_confidence, 0.85) if final_confidence > 0 else 0.0
            except Exception:
                pass

        # eval normalization
        gold_eval = canonicalize_eval_answer(gold, sample)
        baseline_eval_pred = canonicalize_eval_answer(baseline_pred, sample)
        route_eval_pred = canonicalize_eval_answer(route_pred, sample)
        graph_eval_pred = canonicalize_eval_answer(final_pred, sample)

        baseline_exact_eval = exact_match_for_eval(baseline_pred, gold, sample) if baseline_pred else 0
        route_exact_eval = exact_match_for_eval(route_pred, gold, sample) if route_pred else 0
        graph_exact_eval = exact_match_for_eval(final_pred, gold, sample) if final_pred else 0

        baseline_f1_eval = token_f1(baseline_eval_pred, gold_eval) if baseline_pred else 0.0
        route_f1_eval = token_f1(route_eval_pred, gold_eval) if route_pred else 0.0
        graph_f1_eval = token_f1(graph_eval_pred, gold_eval) if final_pred else 0.0

        # evidence metrics
        if route_out.mode in {"lexical_rag", "light_rag"} and route_out.raw_hits:
            hits, evidence_recall, evidence_precision = compute_evidence_metrics(
                route_out.raw_hits,
                retrieval_docs,
                gold_eval,
            )
        else:
            hits, evidence_recall, evidence_precision = compute_serialized_evidence_metrics(
                route_out.evidence,
                gold_eval,
                supporting_docs=retrieval_docs,
            )

        ev2 = compute_evidence_metrics_v2(
            gold=gold,
            pred=final_pred,
            evidence=route_out.evidence,
            sample=sample,
        )

        verification = "Verified" if graph_exact_eval == 1 else "Unverified"

        rec = {
            "sample_key": sample_key,
            "question": q,
            "gold": gold,

            "baseline_pred": baseline_pred,
            "route_pred": route_pred,
            "graph_pred": final_pred,

            "route_answer": route_answer,
            "fallback_answer": fallback_answer,
            "final_answer_source": final_answer_source,

            "baseline_pred_canonical": baseline_eval_pred,
            "route_pred_canonical": route_eval_pred,
            "graph_pred_canonical": graph_eval_pred,
            "gold_canonical": gold_eval,

            "baseline_exact": baseline_exact_eval,
            "route_exact": route_exact_eval,
            "graph_exact": graph_exact_eval,

            "baseline_f1": baseline_f1_eval,
            "route_f1": route_f1_eval,
            "graph_f1": graph_f1_eval,

            "hits": hits,
            "evidence_recall": evidence_recall,
            "evidence_precision": evidence_precision,
            "evidence_metrics_v2": ev2,

            "image_used": bool(image_ref),
            "image_source": image_source,

            "baseline_error": baseline_out.error,
            "graph_error": route_out.error,

            "executor_mode": route_out.mode,
            "decision_status": decision_status,
            "route_confidence": route_confidence,
            "fallback_confidence": fallback_confidence,
            "final_confidence": final_confidence,
            "decision_reason": decision_reason,

            # route 分支原始痕迹
            "tool_logs": route_out.tool_logs,
            "evidence": route_out.evidence,
            "retrieval_trace": route_out.retrieval_trace,
            "kg_trace": route_out.kg_trace,

            # 显式说明最终答案和 trace 的来源
            "final_trace_source": final_trace_source,
            "route_tool_logs": route_out.tool_logs,
            "route_evidence": route_out.evidence,
            "route_retrieval_trace": route_out.retrieval_trace,
            "route_kg_trace": route_out.kg_trace,

            "retrieval_seed_trace": retrieval_seed_trace,

            "difficulty": {
                "overall": diff.overall,
                "qnorm_100": _qnorm_100(diff.overall),
                "bucket": "pending",
                "route": planned_route,
                "task_family": _task_family(sample),
                "linguistic_complexity": diff.linguistic_complexity,
                "reasoning_complexity": diff.reasoning_complexity,
                "multimodal_complexity": diff.multimodal_complexity,
                "knowledge_intensity": diff.knowledge_intensity,
                "choice_ambiguity": diff.choice_ambiguity,
                "retrieval_hardness": diff.retrieval_hardness,
                "deepresearch_coverage": diff.deepresearch_coverage,
            },
            "ablation": {
                "profile": ablation.profile,
                "kg_rag": ablation.use_kg_rag,
                "difficulty_estimator": ablation.use_difficulty,
                "sft_refine": ablation.use_sft,
                "web_search_tool_enabled": bool(effective_web_search_tool_enabled),
                "strict_no_external_kg_when_web_disabled": bool(
                    getattr(settings, "strict_no_external_kg_when_web_disabled", False)
                ),
            },
            "verification": verification,
            "original_record": sample.get("__original_record", {}),
            "normalized_sample": {
                k: _jsonable(v)
                for k, v in sample.items()
                if not k.startswith("__")
            },
        }

        records.append(rec)
        seen_keys.add(sample_key)

        if args.details_out is not None and args.save_every > 0 and len(records) % args.save_every == 0:
            args.details_out.parent.mkdir(parents=True, exist_ok=True)
            if args.details_format == "jsonl":
                with args.details_out.open("w", encoding="utf-8") as f:
                    for item in records:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
            else:
                args.details_out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

        if hasattr(progress, "set_postfix"):
            progress.set_postfix(
                baseline_err=sum(1 for r in records if r["baseline_error"]),
                graph_err=sum(1 for r in records if r["graph_error"]),
                img=sum(1 for r in records if r["image_used"]),
            )

    # bucket stats
    difficulty_values = [float(r.get("difficulty", {}).get("overall", 0.0)) for r in records]
    q1, q2 = _tertile_thresholds(difficulty_values)

    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        overall = float(r.get("difficulty", {}).get("overall", 0.0))
        b = difficulty_bucket(overall, q1=q1, q2=q2)
        r.setdefault("difficulty", {})["bucket"] = b
        r["sft_role"] = _sft_role(b, str(r.get("verification", "Unverified")))
        by_bucket[b].append(r)

    n = max(len(records), 1)

    score_components = {
        "overall": mean([r["difficulty"]["overall"] for r in records]) if records else 0.0,
        "linguistic_complexity": mean([r["difficulty"]["linguistic_complexity"] for r in records]) if records else 0.0,
        "reasoning_complexity": mean([r["difficulty"]["reasoning_complexity"] for r in records]) if records else 0.0,
        "multimodal_complexity": mean([r["difficulty"]["multimodal_complexity"] for r in records]) if records else 0.0,
        "knowledge_intensity": mean([r["difficulty"]["knowledge_intensity"] for r in records]) if records else 0.0,
        "choice_ambiguity": mean([r["difficulty"]["choice_ambiguity"] for r in records]) if records else 0.0,
        "retrieval_hardness": mean([r["difficulty"]["retrieval_hardness"] for r in records]) if records else 0.0,
        "deepresearch_coverage": mean([r["difficulty"]["deepresearch_coverage"] for r in records]) if records else 0.0,
        "image_input_coverage": (sum(1 for r in records if r["image_used"]) / n) if records else 0.0,
        "baseline_error_rate": (sum(1 for r in records if r["baseline_error"]) / n) if records else 0.0,
        "graph_error_rate": (sum(1 for r in records if r["graph_error"]) / n) if records else 0.0,
        "verified_rate": (sum(1 for r in records if r.get("verification") == "Verified") / n) if records else 0.0,
        "avg_tool_calls": (sum(len(r.get("tool_logs", [])) for r in records) / n) if records else 0.0,
        "avg_evidence_count": (sum(len(r.get("evidence", [])) for r in records) / n) if records else 0.0,
        "avg_final_confidence": (sum(_safe_float(r.get("final_confidence", 0.0), 0.0) for r in records) / n) if records else 0.0,
        "abstain_rate": (sum(1 for r in records if r.get("decision_status") == "abstained") / n) if records else 0.0,
    }

    bucket_metrics = {}
    for bucket, items in by_bucket.items():
        m = max(len(items), 1)
        bucket_metrics[bucket] = {
            "count": len(items),
            "baseline_exact": sum(x["baseline_exact"] for x in items) / m,
            "graph_exact": sum(x["graph_exact"] for x in items) / m,
            "baseline_f1": sum(x["baseline_f1"] for x in items) / m,
            "graph_f1": sum(x["graph_f1"] for x in items) / m,
            "avg_difficulty": mean([x["difficulty"]["overall"] for x in items]) if items else 0.0,
            "image_input_coverage": sum(1 for x in items if x["image_used"]) / m,
            "avg_tool_calls": sum(len(x.get("tool_logs", [])) for x in items) / m,
            "avg_evidence_count": sum(len(x.get("evidence", [])) for x in items) / m,
            "avg_final_confidence": sum(_safe_float(x.get("final_confidence", 0.0), 0.0) for x in items) / m,
            "abstain_rate": sum(1 for x in items if x.get("decision_status") == "abstained") / m,
        }

    return {
        "count": len(records),
        "skipped_stats": dict(skipped_stats),
        
        "shard": {
            "num_shards": int(getattr(args, "num_shards", 1) or 1),
            "shard_id": int(getattr(args, "shard_id", 0) or 0),
        },

        "baseline_exact": sum(r["baseline_exact"] for r in records) / n,
        "graph_exact": sum(r["graph_exact"] for r in records) / n,
        "baseline_f1": sum(r["baseline_f1"] for r in records) / n,
        "graph_f1": sum(r["graph_f1"] for r in records) / n,
        "route_exact": sum(r["route_exact"] for r in records) / n,
        "route_f1": sum(r["route_f1"] for r in records) / n,

        "hits": sum(r.get("hits", 0) for r in records) / n,
        "evidence_recall": sum(float(r.get("evidence_recall", 0.0)) for r in records) / n,
        "evidence_precision": sum(float(r.get("evidence_precision", 0.0)) for r in records) / n,

        "difficulty_scores": score_components,
        "difficulty_bucket_thresholds": {"q33": q1, "q67": q2},

        "route_stats": {
            "direct_inference": sum(1 for r in records if r.get("difficulty", {}).get("route") == "direct_inference"),
            "vision_recheck": sum(1 for r in records if r.get("difficulty", {}).get("route") == "vision_recheck"),
            "light_rag": sum(1 for r in records if r.get("difficulty", {}).get("route") == "light_rag"),
            "full_rag": sum(1 for r in records if r.get("difficulty", {}).get("route") == "full_rag"),
        },
        "executor_stats": {
            "direct_inference": sum(1 for r in records if r.get("executor_mode") == "direct_inference"),
            "vision_recheck": sum(1 for r in records if r.get("executor_mode") == "vision_recheck"),
            "light_rag": sum(1 for r in records if r.get("executor_mode") == "light_rag"),
            "lexical_rag": sum(1 for r in records if r.get("executor_mode") == "lexical_rag"),
            "agentic_rag": sum(1 for r in records if r.get("executor_mode") == "agentic_rag"),
            "baseline_short_circuit": sum(1 for r in records if r.get("executor_mode") == "baseline_short_circuit"),
        },
        "decision_stats": {
            "finalized": sum(1 for r in records if r.get("decision_status") == "finalized"),
            "abstained": sum(1 for r in records if r.get("decision_status") == "abstained"),
            "fallback": sum(1 for r in records if r.get("decision_status") == "fallback"),
        },
        "verification_stats": {
            "verified": sum(1 for r in records if r.get("verification") == "Verified"),
            "unverified": sum(1 for r in records if r.get("verification") == "Unverified"),
            "hard_verified": sum(
                1
                for r in records
                if r.get("verification") == "Verified" and r.get("difficulty", {}).get("bucket") == "hard"
            ),
            "hard_unverified": sum(
                1
                for r in records
                if r.get("verification") == "Unverified" and r.get("difficulty", {}).get("bucket") == "hard"
            ),
        },
        "sft_sampling_stats": {
            "hard_unverified_primary": sum(1 for r in records if r.get("sft_role") == "hard_unverified_primary"),
            "hard_verified_aux": sum(1 for r in records if r.get("sft_role") == "hard_verified_aux"),
        },
        "accuracy_stats": {
            "baseline_correct": int(sum(r["baseline_exact"] for r in records)),
            "graph_correct": int(sum(r["graph_exact"] for r in records)),
            "total": int(len(records)),
            "baseline_accuracy": (sum(r["baseline_exact"] for r in records) / n),
            "graph_accuracy": (sum(r["graph_exact"] for r in records) / n),
        },
        "tool_stats": {
            "avg_tool_calls": sum(len(r.get("tool_logs", [])) for r in records) / n,
            "avg_evidence_count": sum(len(r.get("evidence", [])) for r in records) / n,
        },
        "metrics_by_difficulty_bucket": bucket_metrics,
        "details": records,
    }
    
    
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate agentic / lexical-graph enhanced multimodal VQA in DeepResearch")
    parser.add_argument("dataset", type=Path, help="Path to dataset file (.json/.jsonl/.parquet) or dataset directory")
    parser.add_argument("--out", type=Path, default=Path("vqa_eval_report.json"))
    parser.add_argument("--mock", action="store_true", help="Run without LLM API calls")
    parser.add_argument("--model-name", type=str, default=None, help="Override MODEL_NAME for this eval run")
    parser.add_argument("--openai-base-url", type=str, default=None, help="Override OPENAI_BASE_URL for this eval run")
    parser.add_argument("--openai-api-key", type=str, default=None, help="Override OPENAI_API_KEY for this eval run")
    parser.add_argument("--image-root", type=Path, default=None, help="Root folder for relative image paths")
    parser.add_argument("--question-field", type=str, default="question", help="Question column/key name")
    parser.add_argument("--answer-field", type=str, default="answer", help="Answer column/key name")
    parser.add_argument("--documents-field", type=str, default="documents", help="Documents/context column/key name")
    parser.add_argument("--image-url-field", type=str, default="image_url", help="Image URL column/key name")
    parser.add_argument("--image-path-field", type=str, default="image_path", help="Image path column/key name")
    parser.add_argument(
        "--dataset-type",
        choices=["auto", "generic", "aokvqa", "m3cot", "mmmu_pro", "cmmqa", "scienceqa"],
        default="auto",
        help="Dataset parser type.",
    )
    parser.add_argument(
        "--ablation-profile",
        choices=["none", "kg_only", "kg_only1", "difficulty_only", "kg_difficulty", "kg_sft", "difficulty_sft", "all_on"],
        default="all_on",
        help="Ablation assembly profile matching module switches: KG-RAG / difficulty estimator / SFT-refine.",
    )
    parser.add_argument(
        "--mmmu-vision-parquet",
        type=Path,
        default=None,
        help="Optional MMMU-Pro vision parquet file OR directory (joined for separated question/vision parquet setup).",
    )
    parser.add_argument(
        "--mmmu-id-field",
        type=str,
        default="id",
        help="ID field in MMMU-Pro question parquet used to join with --mmmu-vision-parquet.",
    )
    parser.add_argument(
        "--mmmu-vision-id-field",
        type=str,
        default="id",
        help="ID field in MMMU-Pro vision parquet used for join mapping.",
    )
    parser.add_argument(
        "--mmmu-vision-image-field",
        type=str,
        default="image",
        help="Image payload field in MMMU-Pro vision parquet (default: image).",
    )
    parser.add_argument(
        "--details-out",
        type=Path,
        default=None,
        help="Optional path to save per-sample complete records with original fields + eval result (.json or .jsonl)",
    )
    parser.add_argument(
        "--details-format",
        choices=["json", "jsonl"],
        default="json",
        help="Format for --details-out (default: json)",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Resume evaluation from an existing details file (.json/.jsonl). Already completed samples will be skipped.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="If >0 and --details-out is set, periodically checkpoint details every N processed samples.",
    )
    
    parser.add_argument(
        "--num-shards",
        type=int,
        default=int(os.getenv("EVAL_NUM_SHARDS", "1")),
        help="Split evaluation dataset into N deterministic shards for process-level parallel inference.",
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=int(os.getenv("EVAL_SHARD_ID", "0")),
        help="Current shard id, valid range: 0 <= shard_id < num_shards.",
    )
    
    parser.add_argument(
        "--vllm-local-media-root",
        type=Path,
        default=None,
        help="统一的 vLLM 本地媒体允许根目录。所有本地图像与 image_bytes 会被复制/落盘到该目录下再以 file:// 形式传给 vLLM。",
    )
    
    parser.add_argument(
        "--enable-clip-rerank-mllm",
        action="store_true",
        help="Enable independent CLIP retrieval + BGE rerank + MLLM branch.",
    )
    
    parser.add_argument(
        "--enable-self-rag-mllm",
        action="store_true",
        help="Enable independent Self-RAG + MLLM branch. This branch uses local candidate corpus only and does not call web search tools.",
    )
    
    parser.add_argument(
        "--enable-mr2ag-mllm",
        action="store_true",
        help="Enable independent mR2AG + MLLM branch. This branch uses local candidate corpus only and does not call web search tools.",
    )
    

    parser.add_argument(
        "--mr2ag-mode",
        choices=["auto", "no_retrieval", "always_retrieve"],
        default=os.getenv("MR2AG_MLLM_MODE", "auto"),
        help="mR2AG retrieval-reflection policy.",
    )

    parser.add_argument(
        "--mr2ag-retrieval-top-k",
        type=int,
        default=int(os.getenv("MR2AG_RETRIEVAL_TOP_K", "5")),
        help="Default local candidate evidence top-k for mR2AG.",
    )
    
    parser.add_argument(
        "--self-rag-mode",
        choices=["adaptive_retrieval", "no_retrieval", "always_retrieve"],
        default=os.getenv("SELF_RAG_MLLM_MODE", "adaptive_retrieval"),
        help="Self-RAG retrieval policy.",
    )
    
    parser.add_argument(
        "--enable-graphrag-mllm",
        action="store_true",
        help="Enable independent GraphRAG + MLLM branch. This branch uses prebuilt GraphRAG workspace/local corpus only and does not call web search tools.",
    )

    parser.add_argument(
        "--graphrag-mode",
        choices=["local_global", "local", "global", "drift", "basic", "proxy_only"],
        default=os.getenv("GRAPHRAG_MLLM_MODE", "local_global"),
        help="GraphRAG query mode. local_global runs methods from GRAPHRAG_QUERY_METHODS, default local,global.",
    )

    parser.add_argument(
        "--graphrag-root",
        type=Path,
        default=None,
        help="Microsoft GraphRAG workspace root. It should already contain a built GraphRAG index.",
    )

    parser.add_argument(
        "--graphrag-data-dir",
        type=Path,
        default=None,
        help="Optional GraphRAG query --data directory override.",
    )

    parser.add_argument(
        "--graphrag-query-methods",
        type=str,
        default=os.getenv("GRAPHRAG_QUERY_METHODS", "local,global"),
        help="Comma-separated GraphRAG query methods used when --graphrag-mode=local_global.",
    )

    parser.add_argument(
        "--graphrag-require-official",
        action="store_true",
        help="If set, do not fall back to local lexical graph when official GraphRAG CLI/workspace is unavailable.",
    )
    
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=float(os.getenv("EVAL_SAMPLE_RATIO", "1.0")),
        help="Deterministic random sampling ratio before shard split.",
    )

    parser.add_argument(
        "--sample-seed",
        type=int,
        default=int(os.getenv("EVAL_SAMPLE_SEED", "42")),
        help="Random seed for deterministic sample-ratio selection.",
    )
    
    parser.add_argument(
        "--enable-web-search-tool",
        dest="enable_web_search_tool",
        action="store_true",
        default=None,
        help="Allow agentic KG-RAG routes to use search_web/open_webpage/extract_images.",
    )
    parser.add_argument(
        "--disable-web-search-tool",
        dest="enable_web_search_tool",
        action="store_false",
        help="Disable search_web/open_webpage/extract_images in agentic KG-RAG routes.",
    )
    
    args = parser.parse_args()
    
    if args.enable_clip_rerank_mllm:
        os.environ["ENABLE_CLIP_RERANK_MLLM"] = "1"
        settings.enable_clip_rerank_mllm = True
    
    if args.enable_mr2ag_mllm:
        os.environ["ENABLE_MR2AG_MLLM"] = "1"
        settings.enable_mr2ag_mllm = True

    if args.mr2ag_mode:
        os.environ["MR2AG_MLLM_MODE"] = str(args.mr2ag_mode)
        settings.mr2ag_mllm_mode = str(args.mr2ag_mode)

    if args.mr2ag_retrieval_top_k:
        os.environ["MR2AG_RETRIEVAL_TOP_K"] = str(args.mr2ag_retrieval_top_k)
        settings.mr2ag_retrieval_top_k = int(args.mr2ag_retrieval_top_k)

    if args.enable_self_rag_mllm:
        os.environ["ENABLE_SELF_RAG_MLLM"] = "1"
        settings.enable_self_rag_mllm = True

    if args.self_rag_mode:
        os.environ["SELF_RAG_MLLM_MODE"] = str(args.self_rag_mode)
        settings.self_rag_mllm_mode = str(args.self_rag_mode)

    if args.model_name:
        os.environ["MODEL_NAME"] = args.model_name
        settings.model_name = args.model_name

    if args.openai_base_url:
        os.environ["OPENAI_BASE_URL"] = args.openai_base_url
        settings.api_base = args.openai_base_url

    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key
        settings.api_key = args.openai_api_key
    
    if args.enable_graphrag_mllm:
        os.environ["ENABLE_GRAPHRAG_MLLM"] = "1"
        settings.enable_graphrag_mllm = True

    if args.graphrag_mode:
        os.environ["GRAPHRAG_MLLM_MODE"] = str(args.graphrag_mode)
        settings.graphrag_mllm_mode = str(args.graphrag_mode)

    if args.graphrag_root is not None:
        os.environ["GRAPHRAG_ROOT"] = str(args.graphrag_root.expanduser().resolve())
        settings.graphrag_root = os.environ["GRAPHRAG_ROOT"]

    if args.graphrag_data_dir is not None:
        os.environ["GRAPHRAG_DATA_DIR"] = str(args.graphrag_data_dir.expanduser().resolve())
        settings.graphrag_data_dir = os.environ["GRAPHRAG_DATA_DIR"]

    if args.graphrag_query_methods:
        os.environ["GRAPHRAG_QUERY_METHODS"] = str(args.graphrag_query_methods)
        settings.graphrag_query_methods = [
            x.strip().lower()
            for x in str(args.graphrag_query_methods).split(",")
            if x.strip()
        ]

    if args.graphrag_require_official:
        os.environ["GRAPHRAG_REQUIRE_OFFICIAL"] = "1"
        settings.graphrag_require_official = True

    report = run_eval(args.dataset, args=args)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.details_out is not None:
        details = report.get("details", [])
        args.details_out.parent.mkdir(parents=True, exist_ok=True)
        if args.details_format == "jsonl":
            with args.details_out.open("w", encoding="utf-8") as f:
                for item in details:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:
            args.details_out.write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Global Metrics ===")
    print(
        json.dumps(
            {
                "count": report["count"],
                "baseline_exact": report["baseline_exact"],
                "graph_exact": report["graph_exact"],
                "baseline_f1": report["baseline_f1"],
                "graph_f1": report["graph_f1"],
                "hits": report["hits"],
                "evidence_recall": report["evidence_recall"],
                "evidence_precision": report["evidence_precision"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("=== Accuracy Stats ===")
    print(json.dumps(report.get("accuracy_stats", {}), ensure_ascii=False, indent=2))
    print("=== Difficulty Scores ===")
    print(json.dumps(report["difficulty_scores"], ensure_ascii=False, indent=2))
    print("=== Route Stats ===")
    print(json.dumps(report.get("route_stats", {}), ensure_ascii=False, indent=2))
    print("=== Executor Stats ===")
    print(json.dumps(report.get("executor_stats", {}), ensure_ascii=False, indent=2))
    print("=== Verification Stats ===")
    print(json.dumps(report.get("verification_stats", {}), ensure_ascii=False, indent=2))
    print("=== SFT Sampling Stats ===")
    print(json.dumps(report.get("sft_sampling_stats", {}), ensure_ascii=False, indent=2))
    print("=== Tool Stats ===")
    print(json.dumps(report.get("tool_stats", {}), ensure_ascii=False, indent=2))
    print("=== Metrics By Difficulty Bucket ===")
    print(json.dumps(report["metrics_by_difficulty_bucket"], ensure_ascii=False, indent=2))
    print(f"saved: {args.out}")
    if args.details_out is not None:
        print(f"saved details: {args.details_out} ({args.details_format})")


if __name__ == "__main__":
    main()
