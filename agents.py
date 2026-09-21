from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from config import settings
from models import QwenVLClient
from tools import ToolObservation, ToolRouter, parse_loose_json_object
from embeddings import EmbeddingReranker


@dataclass
class Evidence:
    source: str
    claim: str
    excerpt: str
    confidence: float
    supports: list[str] = field(default_factory=list)
    contradicts: list[str] = field(default_factory=list)
    evidence_type: str = "text"  # text / webpage_fact / image_fact / ocr / graph / kb_chunk / hypothesis / ambiguity
    premise_dependencies: list[str] = field(default_factory=list)
    ambiguity: float = 0.0
    exclusivity: float = 0.0


@dataclass
class SubQuestion:
    question: str
    intent: str
    priority: int


@dataclass
class CritiqueResult:
    sufficient: bool
    follow_up_questions: list[str] = field(default_factory=list)
    missing_dimensions: list[str] = field(default_factory=list)
    must_reinspect_image: bool = False
    must_abort_finalization: bool = False
    dominant_failure_mode: str = ""
    final_confidence: float = 0.0
    has_unresolved_contradiction: bool = False


@dataclass
class ResearchState:
    question: str
    sub_questions: list[SubQuestion] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    tool_logs: list[ToolObservation] = field(default_factory=list)


@dataclass
class FinalAnswerDecision:
    answer: str
    confidence: float
    abstain: bool
    reason: str = ""


class PlannerAgent:
    def __init__(self, llm: QwenVLClient) -> None:
        self.llm = llm

    def plan(self, question: str) -> list[SubQuestion]:
        prompt = (
            "将主问题拆解为最多 3 个子问题，并给每个子问题标注 intent 和 priority。"
            '要求输出 JSON 数组，元素格式：{"question":"...","intent":"evidence/counterexample/compare","priority":1}。'
            "如果主问题已经足够明确，可以返回少于 3 个子问题。"
            "如果主问题是选择题（例如包含选项或 A./B./C./D.），优先生成最大化区分候选项的子问题，"
            "而不是先假设某个候选项正确。"
            "每个子问题都尽量同时包含："
            "1) 至少一个候选项语义；"
            "2) 至少一个可观察属性（颜色/形状/位置/关系/OCR/局部区域/动作/环境）。"
            "优先生成："
            "1) 一个用于直接区分最可能两个候选项的子问题；"
            "2) 一个用于排除最强干扰项的子问题；"
            "3) 如仍有必要，再生成一个补充比较或反例子问题。"
            "不要生成纯定义题、百科题、泛常识题。"
            "对图像判断类问题，优先问能够直接由图像可见事实、OCR 或局部区域验证的问题。"
            "尽量避免只围绕模糊候选解释发问。"
        )
        data = self.llm.chat_json(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": question},
            ],
            fallback=[],
            route="planner",
        )

        sub_questions: list[SubQuestion] = []
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                q = str(item.get("question", "")).strip()
                if not q:
                    continue
                try:
                    priority = int(item.get("priority", 3))
                except Exception:
                    priority = 3
                sub_questions.append(
                    SubQuestion(
                        question=q,
                        intent=str(item.get("intent", "evidence")).strip() or "evidence",
                        priority=priority,
                    )
                )

        if not sub_questions:
            sub_questions = [SubQuestion(question=question, intent="evidence", priority=1)]

        return sorted(sub_questions, key=lambda x: x.priority)


class ResearcherAgent:
    """Iterative agent that decides tool actions in ReAct JSON steps."""

    def __init__(self, llm: QwenVLClient, tool_router: ToolRouter) -> None:
        self.llm = llm
        self.tool_router = tool_router
        self.doc_reranker = EmbeddingReranker()
    
    @staticmethod
    def _image_tool_names() -> set[str]:
        return {
            "extract_images",
            "analyze_image",
            "ocr_image",
            "analyze_image_region",
            "ocr_image_region",
        }
    
    
    @staticmethod
    def _web_tool_names() -> set[str]:
        return {
            "search_web",
            "open_webpage",
            "extract_images",
        }

    def _web_search_tool_enabled(self) -> bool:
        return bool(getattr(settings, "enable_web_search_tool", True))

    def _fallback_non_web_action(
        self,
        main_question: str,
        sub_question: SubQuestion,
        logs: list[ToolObservation],
    ) -> dict[str, Any]:
        """
        有图回到本地图像分析；
        无图直接 KG / 本地代理图谱检索，不再先 search_web。
        """
        if self._has_question_image_input():
            return {
                "tool": "analyze_image",
                "args": {
                    "image_ref": "current_image",
                    "question": sub_question.question,
                    "fallback_text": "",
                },
                "reason": "web_search_disabled_use_local_image",
            }

        return self._fallback_kb_retrieval_action_when_no_image(
            main_question=main_question,
            sub_question=sub_question,
            logs=logs,
        )
    
    def postprocess_subquestions(
        self,
        main_question: str,
        sub_questions: list[SubQuestion],
        seed_logs: list[ToolObservation] | None = None,
    ) -> list[SubQuestion]:
        logs = list(seed_logs or [])
        out: list[SubQuestion] = []
        seen_q: set[str] = set()

        for sq in sub_questions:
            candidate = sq

            if not self._is_discriminative_mcq_subquestion(main_question, candidate):
                candidate = self._rewrite_subquestion_from_visual_feedback(
                    main_question=main_question,
                    bad_sub_question=sq,
                    logs=logs,
                    failure_reason="not_discriminative_for_choices",
                    used_questions=list(seen_q),
                )

            if not self._is_discriminative_mcq_subquestion(main_question, candidate):
                continue

            q_norm = str(candidate.question or "").strip()
            if not q_norm or q_norm in seen_q:
                continue

            seen_q.add(q_norm)
            out.append(candidate)

        if out:
            return out

        return [SubQuestion(question=main_question, intent="evidence", priority=1)]
    
    
    def _latest_visual_semantic_context(self, logs: list[ToolObservation]) -> str:
        docs = self._logs_to_documents(logs)
        keep: list[str] = []

        for d in docs:
            src = str(d.get("source", "")).strip().lower()
            txt = str(d.get("text", "")).strip()
            if not txt:
                continue
            if src.startswith((
                "local_image",
                "seed_local_image",
                "current_image",
                "vision_recheck",
                "reinspect:",
            )):
                keep.append(f"[{src}] {txt[:260]}")

        return "\n".join(keep[:4])
    
    
    def _is_option_elimination_subquestion(
        self,
        main_question: str,
        sub_question: SubQuestion,
    ) -> bool:
        main_q = str(main_question or "")
        sub_q = str(getattr(sub_question, "question", "") or "").strip().lower()
        if not sub_q:
            return False

        has_choices = ("选项:" in main_q) or bool(re.search(r"(?:^|\n)\s*[A-H]\.\s+", main_q))
        if not has_choices:
            return False

        option_texts: list[str] = []
        for m in re.finditer(r"(?:^|\n)\s*[A-H]\.\s*(.+)", main_q):
            txt = str(m.group(1)).strip().lower()
            if txt:
                option_texts.append(txt)

        if any(opt and opt in sub_q for opt in option_texts[:10]):
            return True

        cue_phrases = [
            "suggest",
            "indicate",
            "imply",
            "is the area",
            "is this",
            "whether",
            "more likely",
            "rather than",
            "属于",
            "是否",
            "是不是",
            "更像",
            "说明",
            "排除",
            "支持",
            "反例",
        ]
        return any(cue in sub_q for cue in cue_phrases)
    
    
    def _is_discriminative_mcq_subquestion(
        self,
        main_question: str,
        sub_question: SubQuestion,
    ) -> bool:
        main_q = str(main_question or "")
        sub_q = str(getattr(sub_question, "question", "") or "").strip().lower()
        if not sub_q:
            return False

        has_choices = ("选项:" in main_q) or bool(re.search(r"(?:^|\n)\s*[A-H]\.\s+", main_q))
        if not has_choices:
            return True

        option_texts: list[str] = []
        for m in re.finditer(r"(?:^|\n)\s*[A-H]\.\s*(.+)", main_q):
            txt = str(m.group(1)).strip().lower()
            if txt:
                option_texts.append(txt)

        option_hit = any(opt and opt in sub_q for opt in option_texts[:10])

        compare_cues = [
            "rather than", "more likely", "less likely", "distinguish", "compare",
            "排除", "区分", "比较", "反例", "支持", "而不是", "更像", "不是",
        ]
        visual_attr_cues = [
            "color", "shape", "position", "relation", "text", "ocr", "region", "object",
            "animal", "breed", "species", "habitat", "hair", "horn", "face", "body", "road",
            "颜色", "形状", "位置", "关系", "文字", "区域", "物体",
            "动物", "品种", "物种", "栖息地", "毛发", "角", "脸", "身体", "道路",
        ]

        return option_hit or (
            any(c in sub_q for c in compare_cues)
            and any(v in sub_q for v in visual_attr_cues)
        )

    
    def _assess_subquestion_visual_grounding(
        self,
        main_question: str,
        sub_question: SubQuestion,
        evidence: list[Evidence],
        logs: list[ToolObservation],
    ) -> tuple[bool, float, str]:
        if not self._has_question_image_input():
            return True, 1.0, "no_image_input"

        q_terms = self._question_terms_for_retrieval(
            f"{main_question} {sub_question.question}"
        )

        visual_score = 0.0
        overlap_score = 0.0
        ambiguity_penalty = 0.0

        for e in evidence:
            etype = str(getattr(e, "evidence_type", "")).strip().lower()
            source = str(getattr(e, "source", "")).strip().lower()
            merged = f"{getattr(e, 'claim', '')} {getattr(e, 'excerpt', '')}".lower()

            if etype in {"image_fact", "ocr"} or source.startswith(
                ("local_image", "seed_local", "vision_recheck", "reinspect:")
            ):
                visual_score = max(visual_score, float(getattr(e, "confidence", 0.0)))

            overlap = sum(1 for t in q_terms if t in merged)
            if overlap > 0:
                overlap_score = max(
                    overlap_score,
                    min(1.0, overlap / max(len(q_terms), 3)),
                )

            ambiguity_penalty = max(
                ambiguity_penalty,
                float(getattr(e, "ambiguity", 0.0)),
            )

        final_score = 0.55 * visual_score + 0.35 * overlap_score - 0.20 * ambiguity_penalty
        final_score = max(0.0, min(1.0, final_score))

        min_ground = float(getattr(settings, "min_subquestion_grounding_score", 0.55) or 0.55)
        min_visual = float(getattr(settings, "min_visual_grounding_signal", 0.45) or 0.45)

        has_choices = ("选项:" in str(main_question or "")) or bool(
            re.search(r"(?:^|\n)\s*[A-H]\.\s+", str(main_question or ""))
        )

        if has_choices and not self._is_discriminative_mcq_subquestion(main_question, sub_question):
            return False, min(final_score, 0.39), "not_discriminative_for_choices"

        if self._is_option_elimination_subquestion(main_question, sub_question):
            relaxed_ground = max(0.40, min_ground - 0.10)
            relaxed_visual = max(0.40, min_visual - 0.05)

            if visual_score >= relaxed_visual and (
                final_score >= relaxed_ground or overlap_score >= 0.15
            ):
                return True, final_score, "grounded_option_elimination"

        accept = final_score >= min_ground and visual_score >= min_visual

        if accept:
            return True, final_score, "grounded"
        return False, final_score, "not_grounded_to_image"
    
    
    def _rewrite_subquestion_from_visual_feedback(
        self,
        main_question: str,
        bad_sub_question: SubQuestion,
        logs: list[ToolObservation],
        failure_reason: str,
        used_questions: list[str],
    ) -> SubQuestion:
        visual_context = self._latest_visual_semantic_context(logs)
        prompt = (
            "请改写当前失败的子问题。"
            "要求：新问题必须更贴近图像中可验证的视觉特征/OCR/局部区域；"
            "避免继续问泛知识定义；"
            "只输出 JSON 对象："
            '{"question":"...","intent":"evidence","priority":1}'
        )
        data = self.llm.chat_json(
            [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        f"主问题: {main_question}\n"
                        f"失败子问题: {bad_sub_question.question}\n"
                        f"失败原因: {failure_reason}\n"
                        f"已用问题: {used_questions}\n"
                        f"视觉线索:\n{visual_context}"
                    ),
                },
            ],
            fallback={
                "question": bad_sub_question.question,
                "intent": bad_sub_question.intent,
                "priority": bad_sub_question.priority,
            },
            route="reasoning",
        )

        q = str(data.get("question", bad_sub_question.question)).strip() if isinstance(data, dict) else bad_sub_question.question
        if not q or q in used_questions:
            q = bad_sub_question.question

        return SubQuestion(
            question=q,
            intent=str(getattr(bad_sub_question, "intent", "evidence")),
            priority=int(getattr(bad_sub_question, "priority", 1)),
        )

    def _rerank_docs_for_retrieval(
        self,
        question: str,
        docs: list[dict[str, str]],
        keep_k: int | None = None,
    ) -> list[dict[str, str]]:
        if not docs:
            return []

        keep = int(keep_k or getattr(settings, "embedding_rerank_keep_k", 6) or 6)

        filtered = self._filter_docs_for_graph(question, docs)
        if not filtered:
            return []

        reranked = self.doc_reranker.rerank_docs(question, filtered, top_k=max(keep * 2, 1))

        anchors: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()

        def _push(item: dict[str, str], bucket: list[dict[str, str]]) -> None:
            source = str(item.get("source", "")).strip()
            text = str(item.get("text", "")).strip()
            if not text:
                return
            key = (source, text[:200])
            if key in seen:
                return
            seen.add(key)
            bucket.append({"source": source, "text": text, "rerank_score": float(item.get("rerank_score", 0.0) or 0.0)})

        for d in filtered:
            low_source = str(d.get("source", "")).strip().lower()
            if low_source.startswith(("local_image", "seed_local", "current_image", "vision_recheck", "reinspect:", "lexical:local_image")):
                _push(d, anchors)

        merged: list[dict[str, str]] = []
        seen2: set[tuple[str, str]] = set()

        for d in anchors + reranked:
            source = str(d.get("source", "")).strip()
            text = str(d.get("text", "")).strip()
            if not text:
                continue
            key = (source, text[:200])
            if key in seen2:
                continue
            seen2.add(key)
            merged.append(
                {
                    "source": source,
                    "text": text,
                    "rerank_score": float(d.get("rerank_score", 0.0) or 0.0),
                }
            )

        def _bucket_name(source: str) -> str:
            low = str(source or "").strip().lower()
            if low in {"rationale", "reasoning_chains"} or low.startswith("rationale"):
                return "annotation"
            if low.startswith(("local_image", "seed_local", "current_image", "vision_recheck", "reinspect:")):
                return "local_visual"
            if low.startswith("lexical"):
                return "lexical"
            if low.startswith(("http://", "https://")):
                return "web"
            if low in {"dbpedia", "graph_paths", "graph_meta"} or "dbpedia" in low:
                return "graph"
            return "other"

        def _external_quality(d: dict[str, str]) -> float:
            source = str(d.get("source", "")).strip().lower()
            text = str(d.get("text", "")).strip()
            score = float(d.get("rerank_score", 0.0) or 0.0)

            if source.startswith(("http://", "https://")) and text.startswith("搜索结果摘要:"):
                score -= 0.35
            if source in {"dbpedia", "graph_paths", "graph_meta"} or "dbpedia" in source:
                score -= 0.10

            return score

        bucket_caps = {
            "local_visual": 4,
            "annotation": 1,
            "lexical": 2,
            "web": 1,
            "graph": 1,
            "other": 1,
        }

        capped: list[dict[str, str]] = []
        bucket_counts: dict[str, int] = {}

        for d in merged:
            bucket = _bucket_name(d.get("source", ""))
            count = bucket_counts.get(bucket, 0)
            cap = bucket_caps.get(bucket, 2)
            if count >= cap:
                continue
            bucket_counts[bucket] = count + 1
            capped.append({"source": d["source"], "text": d["text"]})
            if len(capped) >= keep:
                break

        reserve_external = any(
            x in str(question or "").lower()
            for x in ["compare", "counterexample", "rather than", "排除", "比较", "区分", "反例", "而不是"]
        )

        if reserve_external and not any(_bucket_name(x.get("source", "")) in {"web", "graph", "other"} for x in capped):
            external_candidates = [
                d for d in merged
                if _bucket_name(d.get("source", "")) in {"web", "graph", "other"}
                and {"source": d["source"], "text": d["text"]} not in capped
            ]
            external_candidates.sort(key=_external_quality, reverse=True)

            if external_candidates:
                best_ext = external_candidates[0]
                if _external_quality(best_ext) >= 0.72:
                    if len(capped) >= keep:
                        for i in range(len(capped) - 1, -1, -1):
                            old_bucket = _bucket_name(capped[i].get("source", ""))
                            if old_bucket in {"annotation", "lexical"}:
                                capped.pop(i)
                                break
                    capped.append({"source": best_ext["source"], "text": best_ext["text"]})

        return capped[:keep]
    
    
    def _has_strong_local_visual_signal(self, logs: list[ToolObservation]) -> bool:
        docs = self._logs_to_documents(logs)

        strong_prefixes = (
            "local_image",
            "seed_local",
            "current_image",
            "vision_recheck",
            "lexical:local_image",
        )

        strong_count = 0
        for d in docs:
            source = str(d.get("source", "")).strip().lower()
            text = str(d.get("text", "")).strip()
            if not text:
                continue
            if source.startswith(strong_prefixes):
                strong_count += 1

        return strong_count >= 2
    
    def _should_block_kg_for_visual_question(
        self,
        main_question: str,
        sub_question: SubQuestion,
        logs: list[ToolObservation],
    ) -> bool:
        if not self._has_question_image_input():
            return False

        q = f"{main_question} {sub_question.question}".lower()
        visual_keywords = [
            "image", "picture", "photo", "screen", "cable", "color", "shape", "position",
            "object", "region", "projector", "laptop",
            "animal", "breed", "species", "object type", "category", "habitat",
            "face", "body", "hair", "horn", "road",
            "图", "图片", "屏幕", "线", "颜色", "形状", "位置", "连接", "区域",
            "动物", "品种", "物种", "类型", "类别", "栖息地",
            "脸", "身体", "毛发", "角", "道路",
        ]

        if not any(k in q for k in visual_keywords):
            return False

        intent = str(getattr(sub_question, "intent", "") or "").strip().lower()
        compare_like = (
            intent in {"compare", "counterexample", "follow_up", "gap_fill"}
            or any(
                x in q
                for x in [
                    "compare", "counterexample", "rather than",
                    "排除", "比较", "区分", "反例", "而不是",
                ]
            )
        )

        docs = self._logs_to_documents(logs)

        has_ambiguity_or_hypothesis = False
        for d in docs:
            source = str(d.get("source", "")).strip().lower()
            text = str(d.get("text", "")).strip().lower()

            if source.endswith("#ambiguity") or source.endswith("#hypothesis"):
                has_ambiguity_or_hypothesis = True
                break

            if any(x in text for x in ["歧义", "ambiguity", "候选解释", "hypothesis"]):
                has_ambiguity_or_hypothesis = True
                break

        # 这些情况需要允许 KG 参与纠偏
        if compare_like or has_ambiguity_or_hypothesis:
            return False

        # 只有视觉信号强 + 没有明显歧义/比较需求时，才阻断 KG
        return self._has_strong_local_visual_signal(logs)

    def _has_question_image_input(self) -> bool:
        return bool(str(getattr(self.tool_router, "current_image_ref", "") or "").strip())

    def _safe_json_loads(self, text: str, fallback: Any = None) -> Any:
        return parse_loose_json_object(text, fallback=fallback)
    
    def _compact_input_data(self, data: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}

        for k, v in (data or {}).items():
            key = str(k)

            if key in {"image_ref", "image_url", "_raw_image_ref"}:
                out[key] = self._compact_ref(str(v))
                continue

            if key == "fallback_text":
                out[key] = str(v)[:240]
                continue

            if key == "graph_context":
                out[key] = str(v)[:400]
                continue

            if key == "documents":
                if isinstance(v, list):
                    out[key] = f"<{len(v)} docs>"
                else:
                    out[key] = "<documents>"
                continue

            if isinstance(v, (str, int, float, bool)) or v is None:
                out[key] = v if not isinstance(v, str) else v[:200]
            else:
                out[key] = str(v)[:200]

        return out
        
    def _compact_ref(self, ref: str) -> str:
        s = str(ref or "").strip()
        if not s:
            return ""

        if s.startswith("data:image/"):
            return "current_image"

        if s.startswith("file://"):
            return "current_image"

        if s.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", s):
            return "current_image"

        if len(s) > 200:
            return s[:200]

        return s
    
    def _fallback_text_action_when_no_image(
        self,
        main_question: str,
        sub_question: SubQuestion,
        logs: list[ToolObservation],
    ) -> dict[str, Any]:
        if not self._web_search_tool_enabled():
            return self._fallback_kb_retrieval_action_when_no_image(
                main_question=main_question,
                sub_question=sub_question,
                logs=logs,
            )

        return {
            "tool": "search_web",
            "args": {
                "query": self._build_search_query(
                    main_question=main_question,
                    sub_question=sub_question,
                    logs=logs,
                ),
                "top_k": 3,
            },
            "reason": "no_image_input_fallback_to_text_retrieval",
        }

    def _clamp01(self, value: Any, default: float = 0.0) -> float:
        try:
            x = float(value)
        except Exception:
            return default
        return max(0.0, min(1.0, x))

    def _compact_search_query(self, text: str) -> str:
        text = str(text or "").strip()

        prefixes = [
            "请帮我", "请你", "请", "帮我", "帮忙",
            "如何", "怎么样", "怎么", "是否", "能否",
            "请分析", "请说明", "请解释", "请判断",
        ]
        for p in prefixes:
            if text.startswith(p):
                text = text[len(p):].strip()

        suffixes = ["？", "?", "吗", "呢", "呀", "吧", "是什么", "是啥"]
        for s in suffixes:
            if text.endswith(s):
                text = text[:-len(s)].strip()

        text = " ".join(text.split())

        if len(text) > 80:
            text = text[:80].strip()

        return text
    
    def _compact_tool_output(self, text: Any, max_chars: int = 320) -> str:
        s = str(text or "").strip()
        if not s:
            return ""
        return s if len(s) <= max_chars else s[:max_chars] + "...<truncated>"


    def _build_tool_history(
        self,
        logs: list[ToolObservation],
        *,
        max_logs: int = 6,
        per_log_chars: int = 320,
        total_chars: int = 2800,
    ) -> str:
        if not logs:
            return ""

        rows: list[str] = []
        used = 0
        recent_logs = logs[-max_logs:]

        for idx, log in enumerate(recent_logs, start=1):
            row = (
                f"{idx}. {log.tool_name} "
                f"输入={self._compact_input_data(log.input_data)} "
                f"输出={self._compact_tool_output(log.output_data, max_chars=per_log_chars)}"
            )
            if used + len(row) > total_chars:
                remain = max(0, total_chars - used)
                if remain > 80:
                    rows.append(row[:remain] + "...<truncated>")
                break

            rows.append(row)
            used += len(row)

        return "\n\n".join(rows)
    

    def _score_search_observation_match(self, question: str, obs: ToolObservation) -> float:
        if obs.tool_name != "search_web":
            return 0.0

        results = self._parse_search_results_from_observation(obs)
        if not results:
            return 0.0

        q_terms = self._question_terms_for_retrieval(question)
        best = 0.0

        for item in results[:3]:
            text = f"{item.get('title', '')} {item.get('snippet', '')}".lower()
            if not text:
                continue
            overlap = sum(1 for t in q_terms if t in text)
            score = overlap / max(len(q_terms), 1)
            best = max(best, min(1.0, score))

        return best


    def _build_search_query(self, main_question: str, sub_question: SubQuestion, logs: list[ToolObservation]) -> str:
        
        if not self._web_search_tool_enabled():
            return self._compact_search_query(sub_question.question)
        base = self._compact_search_query(sub_question.question)

        if bool(getattr(settings, "enable_visual_semantic_web_search", False)):
            visual_context = self._latest_visual_semantic_context(logs)
            if visual_context:
                data = self.llm.chat_json(
                    [
                        {
                            "role": "system",
                            "content": (
                                "请根据子问题和视觉语义线索，生成最多 2 个适合搜索引擎的短 query。"
                                "要求：尽量短；优先实体、对象、场景、行为词；不要输出解释；只输出 JSON 数组。"
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"主问题: {main_question}\n"
                                f"子问题: {sub_question.question}\n"
                                f"视觉线索:\n{visual_context}\n"
                                f"基础query: {base}"
                            ),
                        },
                    ],
                    fallback=[base],
                    route="reasoning",
                )
                if isinstance(data, list) and data:
                    candidate = self._compact_search_query(str(data[0]))
                    if candidate:
                        base = candidate

        recent_domain = ""
        for log in reversed(logs[-4:]):
            if log.tool_name != "open_webpage":
                continue
            recent_url = str(log.input_data.get("url", "")).strip()
            domain = urlparse(recent_url).netloc.strip()
            if domain:
                recent_domain = domain
                break

        if recent_domain and "site:" not in base:
            query = f"{base} site:{recent_domain}"
        else:
            query = base

        return query.strip()
    
    
    def _build_kb_query_variants(
        self,
        main_question: str,
        sub_question: SubQuestion,
        logs: list[ToolObservation],
        max_variants: int = 3,
    ) -> list[str]:
        base = self._build_search_query(
            main_question=main_question,
            sub_question=sub_question,
            logs=logs,
        )
        if not base:
            base = self._compact_search_query(sub_question.question)

        prompt = (
            "请把这个子问题改写成最多 3 个适合知识库检索的短关键词查询。"
            "要求："
            "1) 尽量短，偏实体/概念/过程词；"
            "2) 避免整句口语化问句；"
            "3) 可包含一个偏图像对象词、一个偏机制词、一个偏别名/同义表述；"
            "4) 仅输出 JSON 数组。"
        )
        
        data = self.llm.chat_json(
            [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        f"主问题: {main_question}\n"
                        f"子问题: {sub_question.question}\n"
                        f"基础query: {base}"
                    ),
                },
            ],
            fallback=[base],
            route="reasoning",
        )

        out: list[str] = []
        seen: set[str] = set()

        def _push(x: str) -> None:
            q = self._compact_search_query(x)
            if not q or q in seen:
                return
            seen.add(q)
            out.append(q)

        _push(base)

        if isinstance(data, list):
            for item in data:
                _push(str(item))

        return out[:max_variants]


    def _fallback_kb_retrieval_action_when_no_image(
        self,
        main_question: str,
        sub_question: SubQuestion,
        logs: list[ToolObservation],
    ) -> dict[str, Any]:
        query_variants = self._build_kb_query_variants(
            main_question=main_question,
            sub_question=sub_question,
            logs=logs,
            max_variants=3,
        )

        docs = self._logs_to_documents(logs)
        retrieval_backend = str(getattr(settings, "kg_retrieval_backend", "") or "auto").strip().lower() or "auto"
        
        if (
            not self._web_search_tool_enabled()
            and bool(getattr(settings, "strict_no_external_kg_when_web_disabled", False))
            and retrieval_backend in {"", "auto", "hybrid", "qianfan", "dbpedia"}
        ):
            retrieval_backend = "light_rag"
        
        return {
            "tool": "retrieve_graph_evidence",
            "args": {
                "question": query_variants[0] if query_variants else self._compact_search_query(sub_question.question),
                "query_variants": query_variants,
                "schema": self._proxy_schema_for_question(main_question),
                "graph_context": "\n".join([f"[{d['source']}] {d['text'][:250]}" for d in docs[:6]]),
                "documents": docs[:8],
                "retrieval_backend": retrieval_backend,
                "allow_kb_augmentation": False,
            },
            "reason": f"no_image_input_fallback_to_{retrieval_backend}_kb",
        }
    
    def _extract_image_urls_from_text(self, text: str) -> list[str]:
        if not text:
            return []
        urls = re.findall(r"https?://[^\s\"'<>]+", str(text))
        out: list[str] = []
        seen: set[str] = set()
        for u in urls:
            low = u.lower()
            if not re.search(r"\.(jpg|jpeg|png|webp)(\?|$)", low):
                continue
            if u in seen:
                continue
            seen.add(u)
            out.append(u)
        return out


    def _parse_kb_image_candidates_from_graph_observation(self, obs: ToolObservation) -> list[dict[str, str]]:
        if obs.tool_name != "retrieve_graph_evidence":
            return []

        parsed = self._parse_graph_retrieval_result(obs)
        if not parsed:
            return []

        ge = parsed.get("graph_evidence", {})
        if not isinstance(ge, dict):
            return []

        rows = []
        for key in ["kb_chunks", "query_results"]:
            value = ge.get(key, [])
            if isinstance(value, list):
                rows.extend(value)

        out: list[dict[str, str]] = []
        seen: set[str] = set()

        for row in rows:
            if not isinstance(row, dict):
                continue

            row_source = str(row.get("source", "") or row.get("doc_name", "")).strip().lower()
            if "dbpedia" in row_source:
                continue

            explicit = row.get("image_candidates", [])
            if not isinstance(explicit, list):
                continue

            for u in explicit:
                url = str(u).strip()
                if not url or url in seen:
                    continue
                seen.add(url)
                out.append(
                    {
                        "image_url": url,
                        "alt_text": str(row.get("doc_name", "")).strip(),
                        "title_text": str(row.get("doc_name", "")).strip(),
                        "caption_text": "",
                        "nearby_text": str(row.get("text", ""))[:300],
                    }
                )

        return out[:2]
    
    def _is_visual_text_or_sign_question(self, sub_question: SubQuestion) -> bool:
        q = str(getattr(sub_question, "question", "") or "").lower()
        keywords = [
            "logo", "sign", "label", "poster", "screen", "clock", "timer",
            "text", "ocr", "read", "what does the sign say",
            "color", "shape",
            "标志", "标识", "标签", "海报", "屏幕", "时钟", "计时器",
            "文字", "读出", "ocr", "颜色", "形状",
        ]
        return any(k in q for k in keywords)


    def _should_allow_external_kb_images(
        self,
        sub_question: SubQuestion,
        latest_graph_obs: ToolObservation | None,
    ) -> bool:
        if latest_graph_obs is None:
            return False

        if not self._is_visual_text_or_sign_question(sub_question):
            return False

        parsed = self._parse_graph_retrieval_result(latest_graph_obs)
        if not parsed:
            return False

        ge = parsed.get("graph_evidence", {})
        if not isinstance(ge, dict):
            return False

        rows: list[dict[str, Any]] = []
        for key in ["kb_chunks", "query_results"]:
            value = ge.get(key, [])
            if isinstance(value, list):
                rows.extend([x for x in value if isinstance(x, dict)])

        if not rows:
            return False

        best_score = 0.0
        for row in rows[:5]:
            try:
                score = float(row.get("rerank_score", 0.0) or 0.0)
            except Exception:
                score = 0.0
            best_score = max(best_score, score)

        return best_score >= 0.72


    def _auto_consume_kb_image_candidates(
        self,
        sub_question: SubQuestion,
        logs: list[ToolObservation],
    ) -> list[ToolObservation]:
        new_logs: list[ToolObservation] = []

        latest_graph_obs: ToolObservation | None = None
        for log in reversed(logs):
            if log.tool_name == "retrieve_graph_evidence":
                latest_graph_obs = log
                break

        if latest_graph_obs is None:
            return new_logs

        if not self._should_allow_external_kb_images(
            sub_question=sub_question,
            latest_graph_obs=latest_graph_obs,
        ):
            return new_logs

        candidates = self._parse_kb_image_candidates_from_graph_observation(latest_graph_obs)
        if not candidates:
            return new_logs

        for candidate in candidates[:1]:
            image_url = str(candidate.get("image_url", "")).strip()
            if not image_url:
                continue
            if self._has_image_consumption(logs + new_logs, image_url):
                continue

            fallback_text = self._candidate_to_fallback_text(candidate)

            try:
                analyze_obs = self.tool_router.run(
                    tool_name="analyze_image",
                    args={
                        "image_url": image_url,
                        "question": sub_question.question,
                        "fallback_text": fallback_text,
                        "allow_external_image_ref": True,
                    },
                )
            except Exception as exc:
                analyze_obs = ToolObservation(
                    tool_name="analyze_image",
                    input_data={
                        "image_url": image_url,
                        "question": sub_question.question,
                        "fallback_text": fallback_text,
                        "allow_external_image_ref": True,
                    },
                    output_data=f"TOOL_ERROR: {exc}",
                )
            new_logs.append(analyze_obs)

        return new_logs
    
    def _search_results_count(self, obs: ToolObservation) -> int:
        if obs.tool_name != "search_web":
            return 0
        lines = [ln.strip() for ln in str(obs.output_data or "").splitlines() if ln.strip()]
        count = 0
        for line in lines:
            if re.match(r"^\[\d+\]\s+", line):
                count += 1
        return count


    def _build_search_query_variants(
        self,
        main_question: str,
        sub_question: SubQuestion,
        logs: list[ToolObservation],
        current_obs: ToolObservation | None = None,
        max_variants: int = 3,
    ) -> list[str]:
        
        if not self._web_search_tool_enabled():
            return []
        base = self._build_search_query(
            main_question=main_question,
            sub_question=sub_question,
            logs=logs,
        )
        if not base:
            return []

        visual_context = self._latest_visual_semantic_context(logs)
        failure_context = ""
        if current_obs is not None:
            failure_context = str(current_obs.output_data or "")[:800]

        prompt = (
            "请把当前检索任务改写成最多 3 个搜索 query。"
            "要结合主问题、子问题、视觉线索和上轮低质量结果进行改写。"
            "每个 query 尽量短。"
            "仅输出 JSON 数组。"
        )
        data = self.llm.chat_json(
            [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        f"主问题: {main_question}\n"
                        f"子问题: {sub_question.question}\n"
                        f"基础query: {base}\n"
                        f"视觉线索:\n{visual_context}\n"
                        f"上轮结果:\n{failure_context}"
                    ),
                },
            ],
            fallback=[base],
            route="reasoning",
        )

        out: list[str] = []
        seen: set[str] = set()

        def _push(x: str) -> None:
            q = self._compact_search_query(x)
            if not q or q in seen:
                return
            seen.add(q)
            out.append(q)

        if isinstance(data, list):
            for item in data:
                _push(str(item))

        if not out:
            _push(base)

        return out[:max_variants]

    def _question_terms_for_retrieval(self, text: str) -> set[str]:
        toks = re.findall(r"[A-Za-z0-9_\-\u4e00-\u9fff]+", str(text or "").lower())
        return {t for t in toks if len(t) >= 3}

    def _is_low_quality_search_observation(self, question: str, obs: ToolObservation) -> bool:
        if obs.tool_name != "search_web":
            return False

        lines = [ln.strip() for ln in str(obs.output_data or "").splitlines() if ln.strip()]
        if not lines:
            return True

        q_terms = self._question_terms_for_retrieval(question)
        good_hits = 0

        for line in lines[:3]:
            low = line.lower()
            overlap = sum(1 for t in q_terms if t in low)
            if overlap >= 2:
                good_hits += 1

        return good_hits == 0
    

    def _retry_search_with_variants(
        self,
        main_question: str,
        sub_question: SubQuestion,
        logs: list[ToolObservation],
        previous_obs: ToolObservation,
        top_k: int,
        remaining_budget: int | None = None,
    ) -> list[ToolObservation]:
        
        if not self._web_search_tool_enabled():
            return []
        
        new_logs: list[ToolObservation] = []

        threshold = self._clamp01(
            getattr(settings, "web_search_match_threshold", 0.55),
            default=0.55,
        )
        max_rounds = max(1, int(getattr(settings, "web_search_max_rounds", 5) or 5))

        if remaining_budget is not None:
            try:
                remaining_budget = max(0, int(remaining_budget))
            except Exception:
                remaining_budget = 0

        best_obs = previous_obs
        best_score = self._score_search_observation_match(sub_question.question, previous_obs)

        for _ in range(max_rounds - 1):
            if remaining_budget is not None and remaining_budget <= 0:
                break

            if best_score >= threshold:
                break

            variants = self._build_search_query_variants(
                main_question=main_question,
                sub_question=sub_question,
                logs=logs + new_logs,
                current_obs=best_obs,
                max_variants=3,
            )
            if not variants:
                break

            round_best_obs = None
            round_best_score = best_score

            for q in variants:
                if remaining_budget is not None and remaining_budget <= 0:
                    break

                try:
                    obs = self.tool_router.run(
                        tool_name="search_web",
                        args={"query": q, "top_k": top_k},
                    )
                except Exception as exc:
                    obs = ToolObservation(
                        tool_name="search_web",
                        input_data={"query": q, "top_k": top_k},
                        output_data=f"TOOL_ERROR: {exc}",
                    )

                new_logs.append(obs)

                if remaining_budget is not None:
                    remaining_budget -= 1

                score = self._score_search_observation_match(sub_question.question, obs)

                if score > round_best_score:
                    round_best_score = score
                    round_best_obs = obs

            if round_best_obs is None or round_best_score <= best_score:
                break

            best_obs = round_best_obs
            best_score = round_best_score

        return new_logs
    
    def _parse_search_results_from_observation(self, obs: ToolObservation) -> list[dict[str, str]]:
        if obs.tool_name != "search_web":
            return []

        out: list[dict[str, str]] = []
        lines = [ln.strip() for ln in str(obs.output_data or "").splitlines() if ln.strip()]
        current: dict[str, str] | None = None

        for line in lines:
            one_line = re.match(
                r"^\[(\d+)\]\s+(.*?)\s+\|\s+(https?://\S+)\s+\|\s*(.*)$",
                line,
            )
            if one_line:
                if current:
                    out.append(current)
                    current = None
                out.append(
                    {
                        "title": one_line.group(2).strip(),
                        "link": one_line.group(3).strip(),
                        "snippet": one_line.group(4).strip(),
                    }
                )
                continue

            m = re.match(r"^\[(\d+)\]\s+(.*)$", line)
            if m:
                if current:
                    out.append(current)
                current = {"title": m.group(2).strip(), "link": "", "snippet": ""}
                continue

            if current is None:
                continue

            if line.startswith("URL:"):
                current["link"] = line[len("URL:"):].strip()
            elif line.startswith("Snippet:"):
                current["snippet"] = line[len("Snippet:"):].strip()
            else:
                if current.get("snippet"):
                    current["snippet"] += " " + line
                else:
                    current["snippet"] = line

        if current:
            out.append(current)

        return out


    def _should_open_webpage_from_search(
        self,
        sub_question: SubQuestion,
        obs: ToolObservation,
    ) -> list[str]:
        
        if not self._web_search_tool_enabled():
            return []
        results = self._parse_search_results_from_observation(obs)
        if not results:
            return []

        q_terms = self._question_terms_for_retrieval(sub_question.question)
        scored: list[tuple[str, int]] = []

        for item in results[:3]:
            link = str(item.get("link", "")).strip()
            if not link:
                continue

            text = f"{item.get('title', '')} {item.get('snippet', '')}".lower()
            overlap = sum(1 for t in q_terms if t in text)

            # 仍然保持保守策略：至少 overlap>=2 才开网页
            if overlap >= 2:
                scored.append((link, overlap))

        scored.sort(key=lambda x: x[1], reverse=True)

        out: list[str] = []
        seen: set[str] = set()
        for link, _ in scored[:2]:
            if link in seen:
                continue
            seen.add(link)
            out.append(link)

        return out
    
    def _no_image_kb_escalation_threshold(self) -> int:
        """
        无图情况下，连续多少次低质量文本检索后，才允许升级到 KB。
        这里先做最小实现：默认 2 次。
        """
        return 2


    def _count_recent_low_quality_text_attempts(
        self,
        sub_question: SubQuestion,
        logs: list[ToolObservation],
    ) -> int:
        """
        只统计最近连续的低质量文本检索尝试次数。
        一旦出现一次质量尚可的 search_web / open_webpage，就停止累计。
        """
        count = 0

        for log in reversed(logs):
            tool_name = str(log.tool_name or "").strip()

            if tool_name not in {"search_web", "open_webpage"}:
                if tool_name == "retrieve_graph_evidence":
                    break
                continue

            if tool_name == "open_webpage":
                text = str(log.output_data or "").strip()
                if text and not text.startswith("WEB_FETCH_ERROR:") and not text.startswith("WEB_FETCH_UNSUPPORTED_CONTENT:"):
                    break
                count += 1
                continue

            if tool_name == "search_web":
                if self._search_results_count(log) == 0 or self._is_low_quality_search_observation(
                    sub_question.question,
                    log,
                ):
                    count += 1
                    continue
                break

        return count


    def _should_escalate_no_image_to_kb(
        self,
        sub_question: SubQuestion,
        logs: list[ToolObservation],
    ) -> bool:
        return self._count_recent_low_quality_text_attempts(
            sub_question=sub_question,
            logs=logs,
        ) >= self._no_image_kb_escalation_threshold()
    

    def _proxy_schema_for_question(self, main_question: str) -> dict[str, Any]:
        return {
            "summary": f"Question-driven proxy schema for: {main_question}",
            "node_types": [
                "Entity",
                "Concept",
                "WebPage",
                "ImageObject",
                "ImageDescription",
                "OCRText",
                "KnowledgeChunk",
            ],
            "relation_types": [
                "mentions",
                "depicts",
                "located_in",
                "related_to",
                "supports",
                "contradicts",
                "extracted_from",
                "described_by",
                "contains_text",
                "about",
                "same_as",
            ],
        }


    def _next_action(self, main_question: str, sub_question: SubQuestion, logs: list[ToolObservation]) -> dict:
        tool_history = self._build_tool_history(
            logs,
            max_logs=6,
            per_log_chars=320,
            total_chars=2800,
        )

        has_question_image = self._has_question_image_input()
        web_search_enabled = self._web_search_tool_enabled()

        image_rule = (
            "- 当前问题没有图像输入；禁止选择 extract_images / analyze_image / ocr_image / analyze_image_region / ocr_image_region。\n"
            if not has_question_image else
            "- 当前问题有图像输入；只有在确实需要视觉证据时才调用图像工具。\n"
        )

        if web_search_enabled:
            tool_catalog = (
                "1) search_web {query, top_k}\n"
                "2) open_webpage {url}\n"
                "3) extract_images {url}\n"
                "4) analyze_image {image_url 或 image_ref, question, fallback_text}\n"
                "5) ocr_image {image_url 或 image_ref, fallback_text}\n"
                "6) analyze_image_region {image_url 或 image_ref, question, region_hint, fallback_text}\n"
                "7) ocr_image_region {image_url 或 image_ref, region_hint, fallback_text}\n"
                "8) retrieve_graph_evidence {question, schema, graph_context, retrieval_backend?}\n"
                "9) finish {note}\n"
            )
            web_rule = (
                "- search_web 返回的是摘要级证据，不等于完整正文；只有在需要核实细节时再 open_webpage；\n"
                "- 当前问题没有图像输入时，禁止编造或猜测任何 image_url / image_ref；若需要外部信息，先用 search_web / open_webpage，只有在连续多次文本检索质量仍低时才考虑 retrieve_graph_evidence。\n"
            )
            fallback_action = {
                "tool": "search_web",
                "args": {
                    "query": self._build_search_query(
                        main_question=main_question,
                        sub_question=sub_question,
                        logs=logs,
                    ),
                    "top_k": 5,
                },
                "reason": "fallback",
            }
        else:
            tool_catalog = (
                "1) analyze_image {image_url 或 image_ref, question, fallback_text}\n"
                "2) ocr_image {image_url 或 image_ref, fallback_text}\n"
                "3) analyze_image_region {image_url 或 image_ref, question, region_hint, fallback_text}\n"
                "4) ocr_image_region {image_url 或 image_ref, region_hint, fallback_text}\n"
                "5) retrieve_graph_evidence {question, schema, graph_context, retrieval_backend?}\n"
                "6) finish {note}\n"
            )
            web_rule = (
                "- 当前 ENABLE_WEB_SEARCH_TOOL=0；严禁选择 search_web / open_webpage / extract_images；\n"
                "- 不要生成搜索引擎 query，不要请求打开网页，不要提取网页图片候选；\n"
                "- 如果需要外部知识，只能基于已有本地图像、本地文档、题目文本和 retrieve_graph_evidence 的 KG / 本地代理图谱证据进行推理；\n"
                "- 当前问题没有图像输入时，不要先尝试 web search；若需要补充知识，直接考虑 retrieve_graph_evidence 或 finish。\n"
            )
            fallback_action = self._fallback_non_web_action(
                main_question=main_question,
                sub_question=sub_question,
                logs=logs,
            )

        prompt = (
            "假设你是一个专业的精简研究 Agent。你必须从以下工具中选择一个行动：\n"
            f"{tool_catalog}"
            "规则：\n"
            f"{image_rule}"
            "- 优先少步完成，通常总共不超过4步；\n"
            "- 对图像判断类问题，优先视觉工具，不要先使用知识图谱；\n"
            "- lexical graph 由系统在检索路线中自动补充，不需要你主动选择；\n"
            "- 如果当前主要证据已经来自本地图像/本地文档，且足以区分竞争解释，不要再调用 retrieve_graph_evidence；\n"
            f"{web_rule}"
            "- 当问题偏事实、常识、百科、多跳知识或背景解释，而当前证据仍不足时，可调用 retrieve_graph_evidence；\n"
            "- 当你已经有足够证据回答当前子问题时，选择 finish；\n"
            '仅输出 JSON 对象，格式：{"tool":"retrieve_graph_evidence","args":{...},"reason":"..."}'
        )

        action = self.llm.chat_json(
            [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        f"主问题: {main_question}\n"
                        f"当前子问题: {sub_question.question}\n"
                        f"子问题意图: {sub_question.intent}\n"
                        f"已有工具观察:\n{tool_history or '无'}"
                    ),
                },
            ],
            fallback=fallback_action,
            route="reasoning",
        )

        if not isinstance(action, dict):
            action = dict(fallback_action)
            action["reason"] = "invalid_action_fallback"

        tool_name = str(action.get("tool", "")).strip()
        args = action.get("args", {})
        if not isinstance(args, dict):
            args = {}
        normalized_route = str(args.get("route_mode", "") or args.get("retrieval_mode", "") or "").strip().lower()

        # web search tool 关闭时，模型即使误选也要重路由。
        if not web_search_enabled and tool_name in self._web_tool_names():
            return self._fallback_non_web_action(
                main_question=main_question,
                sub_question=sub_question,
                logs=logs,
            )

        # 视觉问题：若当前还没有足够视觉锚点，不要先 search_web，先回到图像
        if tool_name == "search_web" and not self._should_search_web_for_subquestion(sub_question, logs):
            return {
                "tool": "analyze_image",
                "args": {
                    "image_ref": "current_image",
                    "question": sub_question.question,
                    "fallback_text": "",
                },
                "reason": "need_visual_grounding_before_web_search",
            }

        # 两阶段 no-image fallback：
        # web search 开启时，先文本检索，连续低质量后再 KG；
        # web search 关闭时，直接 KG。
        if not has_question_image:
            if tool_name in self._image_tool_names():
                if (not web_search_enabled) or self._should_escalate_no_image_to_kb(
                    sub_question=sub_question,
                    logs=logs,
                ):
                    return self._fallback_kb_retrieval_action_when_no_image(
                        main_question=main_question,
                        sub_question=sub_question,
                        logs=logs,
                    )
                return self._fallback_text_action_when_no_image(
                    main_question=main_question,
                    sub_question=sub_question,
                    logs=logs,
                )

            if tool_name == "retrieve_graph_evidence":
                if web_search_enabled and not self._should_escalate_no_image_to_kb(
                    sub_question=sub_question,
                    logs=logs,
                ):
                    return self._fallback_text_action_when_no_image(
                        main_question=main_question,
                        sub_question=sub_question,
                        logs=logs,
                    )

        if tool_name == "retrieve_graph_evidence" and self._should_block_kg_for_visual_question(
            main_question=main_question,
            sub_question=sub_question,
            logs=logs,
        ):
            logs.append(
                ToolObservation(
                    tool_name="kg_gating",
                    input_data={
                        "main_question": main_question,
                        "sub_question": sub_question.question,
                        "route_mode": normalized_route,
                    },
                    output_data=json.dumps(
                        {
                            "reason": "blocked_by_strong_visual_signal",
                            "tool": "retrieve_graph_evidence",
                        },
                        ensure_ascii=False,
                    ),
                )
            )
            return {
                "tool": "finish",
                "args": {"note": "visual_evidence_already_sufficient_skip_kg"},
                "reason": "skip_kg_for_visual_question",
            }

        if tool_name == "search_web":
            raw_query = args.get("query", sub_question.question)
            args["query"] = self._build_search_query(
                main_question=main_question,
                sub_question=SubQuestion(
                    question=str(raw_query),
                    intent=sub_question.intent,
                    priority=sub_question.priority,
                ),
                logs=logs,
            )
            try:
                args["top_k"] = min(max(1, int(args.get("top_k", 5))), 6)
            except Exception:
                args["top_k"] = 5
            action["args"] = args

        elif tool_name == "retrieve_graph_evidence":
            args.setdefault("question", sub_question.question)
            args.setdefault("schema", self._proxy_schema_for_question(main_question))
            if "graph_context" not in args:
                docs = self._logs_to_documents(logs)
                args["graph_context"] = "\n".join(
                    [f"[{d['source']}] {d['text'][:250]}" for d in docs[:6]]
                )

            if not web_search_enabled:
                args.setdefault("allow_kb_augmentation", False)
                if bool(getattr(settings, "strict_no_external_kg_when_web_disabled", False)):
                    args["retrieval_backend"] = "light_rag"

            action["args"] = args

        elif tool_name in self._image_tool_names():
            if tool_name in {"analyze_image", "analyze_image_region"}:
                args.setdefault("question", sub_question.question)
            args.setdefault("fallback_text", "")
            action["args"] = args

        return action
    
    
    def _is_error_observation(self, log: ToolObservation) -> bool:
        return self._is_non_evidence_observation(log)

    def _is_non_evidence_observation(self, log: ToolObservation) -> bool:
        text = str(log.output_data or "")
        prefixes = (
            "TOOL_ERROR:",
            "WEB_FETCH_ERROR:",
            "WEB_FETCH_UNSUPPORTED_CONTENT:",
            "SKIPPED_",
            "STOPPED_LOW_VALUE_",
        )
        return any(text.startswith(p) for p in prefixes)

    def _parse_image_candidates_from_observation(self, obs: ToolObservation) -> list[dict[str, str]]:
        if obs.tool_name != "extract_images":
            return []

        candidates: list[dict[str, str]] = []
        for line in str(obs.output_data or "").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            if not isinstance(item, dict):
                continue
            image_url = str(item.get("image_url", "")).strip()
            if not image_url:
                continue
            candidates.append(
                {
                    "image_url": image_url,
                    "alt_text": str(item.get("alt_text", "")).strip(),
                    "title_text": str(item.get("title_text", "")).strip(),
                    "caption_text": str(item.get("caption_text", "")).strip(),
                    "nearby_text": str(item.get("nearby_text", "")).strip(),
                }
            )
        return candidates

    def _candidate_to_fallback_text(self, candidate: dict[str, str]) -> str:
        parts: list[str] = []
        for key in ["alt_text", "title_text", "caption_text", "nearby_text"]:
            value = str(candidate.get(key, "")).strip()
            if value:
                parts.append(f"{key}: {value}")
        if not parts and candidate.get("image_url"):
            parts.append(f"image_url: {candidate['image_url']}")
        return " | ".join(parts)[:1000]

    def _has_image_consumption(self, logs: list[ToolObservation], image_url: str) -> bool:
        for log in logs:
            if log.tool_name not in {"analyze_image", "ocr_image", "analyze_image_region", "ocr_image_region"}:
                continue
            used = str(log.input_data.get("image_url", "") or log.input_data.get("image_ref", "")).strip()
            if used == image_url:
                return True
        return False

    def _should_try_ocr(self, candidate: dict[str, str]) -> bool:
        text = " ".join(
            [
                str(candidate.get("alt_text", "")),
                str(candidate.get("title_text", "")),
                str(candidate.get("caption_text", "")),
                str(candidate.get("nearby_text", "")),
            ]
        ).lower()

        keywords = [
            "text", "ocr", "chart", "diagram", "table", "figure", "screenshot", "poster", "slide", "map",
            "文字", "图表", "表格", "示意图", "截图", "海报", "坐标", "标注", "地图", "clock", "time", "timer", "watch",
        ]
        return any(k in text for k in keywords)

    def _parse_structured_image_analysis(self, obs: ToolObservation) -> dict[str, Any]:
        if obs.tool_name not in {"analyze_image", "analyze_image_region"}:
            return {}
        parsed = self._safe_json_loads(str(obs.output_data or ""), fallback={})
        return parsed if isinstance(parsed, dict) else {}

    def _parse_structured_ocr(self, obs: ToolObservation) -> dict[str, Any]:
        if obs.tool_name not in {"ocr_image", "ocr_image_region"}:
            return {}
        parsed = self._safe_json_loads(str(obs.output_data or ""), fallback={})
        return parsed if isinstance(parsed, dict) else {}

    def _parse_graph_retrieval_result(self, obs: ToolObservation) -> dict[str, Any]:
        if obs.tool_name != "retrieve_graph_evidence":
            return {}
        parsed = self._safe_json_loads(str(obs.output_data or ""), fallback={})
        return parsed if isinstance(parsed, dict) else {}

    def _normalize_region_hint(self, hint: str) -> str:
        h = str(hint or "").strip().lower()
        mapping = {
            "inspect_clock_region": "clock",
            "inspect_window_or_outdoor": "window",
            "inspect_light_source": "light_source",
            "inspect_candles": "candles",
            "clock": "clock",
            "window": "window",
            "light_source": "light_source",
            "candles": "candles",
            "sign": "sign",
            "screen": "screen",
            "poster": "poster",
            "table": "table",
        }
        return mapping.get(h, h or "relevant_region")

    def _auto_consume_image_candidates(
        self,
        sub_question: SubQuestion,
        logs: list[ToolObservation],
    ) -> list[ToolObservation]:
        new_logs: list[ToolObservation] = []

        if not self._has_question_image_input():
            return new_logs

        latest_candidates: list[dict[str, str]] = []
        for log in reversed(logs):
            if log.tool_name == "extract_images":
                latest_candidates = self._parse_image_candidates_from_observation(log)
                if latest_candidates:
                    break

        if not latest_candidates:
            return new_logs

        for candidate in latest_candidates[:2]:
            image_url = str(candidate.get("image_url", "")).strip()
            if not image_url:
                continue
            if self._has_image_consumption(logs + new_logs, image_url):
                continue

            fallback_text = self._candidate_to_fallback_text(candidate)

            try:
                analyze_obs = self.tool_router.run(
                    tool_name="analyze_image",
                    args={
                        "image_url": image_url,
                        "question": sub_question.question,
                        "fallback_text": fallback_text,
                    },
                )
            except Exception as exc:
                analyze_obs = ToolObservation(
                    tool_name="analyze_image",
                    input_data={"image_url": image_url, "question": sub_question.question, "fallback_text": fallback_text},
                    output_data=f"TOOL_ERROR: {exc}",
                )
            new_logs.append(analyze_obs)

            analysis = self._parse_structured_image_analysis(analyze_obs)
            followups = analysis.get("suggested_followups", []) if isinstance(analysis, dict) else []
            if not isinstance(followups, list):
                followups = []

            consumed_regions: set[str] = set()
            for hint in followups[:2]:
                region_hint = self._normalize_region_hint(str(hint))
                if not region_hint or region_hint in consumed_regions:
                    continue
                consumed_regions.add(region_hint)

                try:
                    region_obs = self.tool_router.run(
                        tool_name="analyze_image_region",
                        args={
                            "image_url": image_url,
                            "question": sub_question.question,
                            "region_hint": region_hint,
                            "fallback_text": fallback_text,
                        },
                    )
                except Exception as exc:
                    region_obs = ToolObservation(
                        tool_name="analyze_image_region",
                        input_data={
                            "image_url": image_url,
                            "question": sub_question.question,
                            "region_hint": region_hint,
                            "fallback_text": fallback_text,
                        },
                        output_data=f"TOOL_ERROR: {exc}",
                    )
                new_logs.append(region_obs)

                if region_hint in {"clock", "sign", "screen", "poster", "table"} or self._should_try_ocr(candidate):
                    try:
                        ocr_region_obs = self.tool_router.run(
                            tool_name="ocr_image_region",
                            args={
                                "image_url": image_url,
                                "region_hint": region_hint,
                                "fallback_text": fallback_text,
                            },
                        )
                    except Exception as exc:
                        ocr_region_obs = ToolObservation(
                            tool_name="ocr_image_region",
                            input_data={
                                "image_url": image_url,
                                "region_hint": region_hint,
                                "fallback_text": fallback_text,
                            },
                            output_data=f"TOOL_ERROR: {exc}",
                        )
                    new_logs.append(ocr_region_obs)

            if self._should_try_ocr(candidate):
                try:
                    ocr_obs = self.tool_router.run(
                        tool_name="ocr_image",
                        args={
                            "image_url": image_url,
                            "fallback_text": fallback_text,
                        },
                    )
                except Exception as exc:
                    ocr_obs = ToolObservation(
                        tool_name="ocr_image",
                        input_data={"image_url": image_url, "fallback_text": fallback_text},
                        output_data=f"TOOL_ERROR: {exc}",
                    )
                new_logs.append(ocr_obs)

        return new_logs

    def _logs_to_documents(self, logs: list[ToolObservation]) -> list[dict[str, str]]:
        docs: list[dict[str, str]] = []

        for lg in logs:
            if self._is_non_evidence_observation(lg):
                continue
            
            if (not self._web_search_tool_enabled()) and lg.tool_name in self._web_tool_names():
                continue

            raw_output = str(lg.output_data or "").strip()
            if raw_output == "SKIPPED_NO_IMAGE_INPUT":
                continue

            if lg.tool_name == "search_web":
                for item in self._parse_search_results_from_observation(lg)[:3]:
                    url = str(item.get("link", "")).strip()
                    title = str(item.get("title", "")).strip()
                    snippet = str(item.get("snippet", "")).strip()
                    text = "\n".join([x for x in [title, snippet] if x]).strip()
                    if not url or not text:
                        continue
                    docs.append(
                        {
                            "source": f"{url}#search_snippet",
                            "text": f"搜索结果摘要(未打开网页核验):\n{text[:1200]}",
                        }
                    )
                continue

            if lg.tool_name == "open_webpage":
                url = str(lg.input_data.get("url", "unknown"))
                text = raw_output
                if text:
                    docs.append({"source": url, "text": text})

            elif lg.tool_name == "search_web":
                lines = [ln.strip() for ln in raw_output.splitlines() if ln.strip()]
                for line in lines[:3]:
                    parts = [p.strip() for p in line.split(" | ", 2)]
                    if len(parts) < 2:
                        continue

                    title = re.sub(r"^\[\d+\]\s*", "", parts[0]).strip()
                    url = parts[1].strip()
                    snippet = parts[2].strip() if len(parts) >= 3 else ""

                    if not url:
                        continue

                    text = "\n".join([x for x in [title, snippet] if x]).strip()
                    if not text:
                        continue

                    docs.append(
                        {
                            "source": url,
                            "text": f"搜索结果摘要:\n{text[:1200]}",
                        }
                    )

            elif lg.tool_name in {"analyze_image", "analyze_image_region"}:
                parsed = self._parse_structured_image_analysis(lg)
                if isinstance(parsed, dict) and parsed.get("skipped") is True:
                    continue

                image_src = self._compact_ref(
                    str(lg.input_data.get("image_url", "") or lg.input_data.get("image_ref", "")).strip()
                ) or "image_analysis"

                if isinstance(parsed, dict):
                    visible_facts = parsed.get("visible_facts", {})
                    if isinstance(visible_facts, dict) and visible_facts:
                        fact_lines = [f"{k}: {v}" for k, v in visible_facts.items()]
                        docs.append({"source": image_src, "text": "图像可见事实:\n" + "\n".join(fact_lines[:12])})

                    ocr_facts = parsed.get("ocr_facts", [])
                    if isinstance(ocr_facts, list) and ocr_facts:
                        ocr_lines = []
                        for item in ocr_facts[:10]:
                            if not isinstance(item, dict):
                                continue
                            text = str(item.get("text", "")).strip()
                            region = str(item.get("region", "unknown")).strip()
                            conf = item.get("confidence", "")
                            if text:
                                ocr_lines.append(f"[{region}] {text} (confidence={conf})")
                        if ocr_lines:
                            docs.append({"source": f"{image_src}#ocr", "text": "图像OCR事实:\n" + "\n".join(ocr_lines)})

                    hypotheses = parsed.get("hypotheses", [])
                    if isinstance(hypotheses, list) and hypotheses:
                        hyp_lines = []
                        for item in hypotheses[:6]:
                            if not isinstance(item, dict):
                                continue
                            label = str(item.get("label", "")).strip()
                            reason = str(item.get("reason", "")).strip()
                            conf = item.get("confidence", "")
                            if label or reason:
                                hyp_lines.append(f"{label} | reason={reason} | confidence={conf}")
                        if hyp_lines:
                            docs.append({"source": f"{image_src}#hypothesis", "text": "图像候选解释:\n" + "\n".join(hyp_lines)})

            elif lg.tool_name in {"ocr_image", "ocr_image_region"}:
                parsed = self._parse_structured_ocr(lg)
                if isinstance(parsed, dict) and parsed.get("skipped") is True:
                    continue

                image_src = self._compact_ref(
                    str(lg.input_data.get("image_url", "") or lg.input_data.get("image_ref", "")).strip()
                ) or "ocr"

                if isinstance(parsed, dict):
                    ocr_lines = parsed.get("ocr_lines", [])
                    if isinstance(ocr_lines, list) and ocr_lines:
                        lines = []
                        for item in ocr_lines[:12]:
                            if not isinstance(item, dict):
                                continue
                            text = str(item.get("text", "")).strip()
                            region = str(item.get("region", "unknown")).strip()
                            conf = item.get("confidence", "")
                            if text:
                                lines.append(f"[{region}] {text} (confidence={conf})")
                        if lines:
                            docs.append({"source": f"{image_src}#ocr", "text": "OCR提取文本:\n" + "\n".join(lines)})

            elif lg.tool_name == "retrieve_lexical_graph":
                text = raw_output
                if text:
                    docs.append({"source": "lexical_graph", "text": text[:3000]})

            elif lg.tool_name == "retrieve_graph_evidence":
                parsed = self._parse_graph_retrieval_result(lg)
                if not isinstance(parsed, dict):
                    continue
                ge = parsed.get("graph_evidence", {})
                if not isinstance(ge, dict):
                    continue
                la = parsed.get("linking_artifacts", {})
                if not isinstance(la, dict):
                    la = {}
                provider_confidence = self._clamp01(parsed.get("confidence", 0.0), default=0.0)
                provider_coverage = self._clamp01(parsed.get("coverage", 0.0), default=0.0)
                backend = str(
                    la.get("effective_backend", "")
                    or la.get("external_backend", "")
                    or lg.input_data.get("_resolved_backend", "")
                    or ""
                ).strip()
                selected_strategy = str(la.get("selected_strategy", "")).strip()

                for row in ge.get("query_results", [])[:8]:
                    if not isinstance(row, dict):
                        continue
                    source = str(row.get("source", "") or row.get("doc_name", "") or "proxy_doc").strip()
                    text = str(row.get("text", "")).strip()
                    if text:
                        docs.append(
                            {
                                "source": source,
                                "text": text[:1200],
                                "rerank_score": row.get("rerank_score", row.get("score", 0.0)),
                                "provider_confidence": provider_confidence,
                                "provider_coverage": provider_coverage,
                                "kg_backend": backend,
                                "selected_strategy": selected_strategy,
                                "evidence_origin": row.get("evidence_origin", la.get("external_backend", "")),
                                "relation_is_inferred": bool(row.get("relation_is_inferred", False)),
                            }
                        )

        return docs
    
    
    def _dedupe_evidence(self, evidence: list[Evidence], limit: int = 8) -> list[Evidence]:
        out: list[Evidence] = []
        seen: set[tuple[str, str, str, str]] = set()

        for e in evidence:
            key = (
                str(getattr(e, "source", "")).strip(),
                str(getattr(e, "claim", "")).strip(),
                str(getattr(e, "excerpt", "")).strip()[:240],
                str(getattr(e, "evidence_type", "")).strip(),
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(e)

        return out[:limit]

    def _lift_documents_to_evidence(self, docs: list[dict[str, str]]) -> list[Evidence]:
        out: list[Evidence] = []

        for d in docs[:8]:
            if not isinstance(d, dict):
                continue

            source = str(d.get("source", "")).strip() or "unknown"
            text = str(d.get("text", "")).strip()
            if not text:
                continue

            low_source = source.lower()
            low_text = text.lower()
            kg_backend = str(d.get("kg_backend", "")).strip()
            selected_strategy = str(d.get("selected_strategy", "")).strip()
            evidence_origin = str(d.get("evidence_origin", "")).strip()
            is_kg_doc = bool(kg_backend or selected_strategy or evidence_origin) or "dbpedia" in low_source
            row_score = self._clamp01(d.get("rerank_score", d.get("score", 0.0)), default=0.0)
            provider_confidence = self._clamp01(d.get("provider_confidence", 0.0), default=0.0)
            provider_coverage = self._clamp01(d.get("provider_coverage", 0.0), default=0.0)
            kg_quality_confidence = max(
                0.30,
                min(0.85, 0.50 * row_score + 0.35 * provider_confidence + 0.15 * provider_coverage),
            )

            if low_source.endswith("#search_snippet"):
                out.append(
                    Evidence(
                        source=source,
                        claim="搜索结果摘要",
                        excerpt=text[:600],
                        confidence=0.50,
                        evidence_type="search_snippet",
                        premise_dependencies=["仅来自搜索摘要，未打开网页正文核验"],
                        ambiguity=0.30,
                        exclusivity=0.12,
                    )
                )
                continue

            if low_source.endswith("#ambiguity"):
                out.append(
                    Evidence(
                        source=source,
                        claim="存在歧义或未决因素",
                        excerpt=text[:600],
                        confidence=0.62,
                        evidence_type="ambiguity",
                        premise_dependencies=[],
                        ambiguity=0.95,
                        exclusivity=0.0,
                    )
                )
                continue

            if low_source.endswith("#hypothesis"):
                out.append(
                    Evidence(
                        source=source,
                        claim=source.split("#")[-1] or "候选解释",
                        excerpt=text[:600],
                        confidence=0.55,
                        evidence_type="hypothesis",
                        premise_dependencies=["需要与直接观测证据交叉验证"],
                        ambiguity=0.40,
                        exclusivity=0.18,
                    )
                )
                continue

            if low_source.endswith("#ocr") or low_text.startswith("ocr提取文本") or low_text.startswith("图像ocr事实"):
                out.append(
                    Evidence(
                        source=source,
                        claim="OCR事实",
                        excerpt=text[:600],
                        confidence=0.80,
                        evidence_type="ocr",
                        premise_dependencies=[],
                        ambiguity=0.08,
                        exclusivity=0.45,
                    )
                )
                continue

            if low_source.startswith(("local_image", "seed_local_image", "current_image", "vision_recheck", "reinspect:")):
                out.append(
                    Evidence(
                        source=source,
                        claim="图像可见事实",
                        excerpt=text[:600],
                        confidence=0.76,
                        evidence_type="image_fact",
                        premise_dependencies=[],
                        ambiguity=0.06,
                        exclusivity=0.36,
                    )
                )
                continue

            if low_source in {"lexical_graph", "graph_paths", "graph_meta"} or is_kg_doc:
                out.append(
                    Evidence(
                        source=source,
                        claim="图谱/检索结构证据",
                        excerpt=text[:600],
                        confidence=kg_quality_confidence,
                        evidence_type="kb_chunk" if ("qianfan" in low_source or kg_backend == "qianfan") else "graph",
                        premise_dependencies=[
                            x for x in [
                                f"backend={kg_backend}" if kg_backend else "",
                                f"strategy={selected_strategy}" if selected_strategy else "",
                                "relation_is_inferred" if d.get("relation_is_inferred") else "",
                            ]
                            if x
                        ],
                        ambiguity=0.30 if d.get("relation_is_inferred") else 0.16,
                        exclusivity=0.30,
                    )
                )
                continue

            if low_source.startswith(("http://", "https://")):
                out.append(
                    Evidence(
                        source=source,
                        claim="网页正文事实",
                        excerpt=text[:600],
                        confidence=0.68,
                        evidence_type="webpage_fact",
                        premise_dependencies=[],
                        ambiguity=0.18,
                        exclusivity=0.26,
                    )
                )
                continue

            out.append(
                Evidence(
                    source=source,
                    claim="文本证据",
                    excerpt=text[:600],
                    confidence=0.62,
                    evidence_type="text",
                    premise_dependencies=[],
                    ambiguity=0.22,
                    exclusivity=0.20,
                )
            )

        return self._dedupe_evidence(out, limit=8)
    
    
    def _distill_evidence(self, main_question: str, sub_question: str, logs: list[ToolObservation]) -> list[Evidence]:
        useful_logs = [log for log in logs if not self._is_non_evidence_observation(log)]
        compact_logs = "\n\n".join(
            [f"工具={log.tool_name}\n输入={log.input_data}\n输出={str(log.output_data)[:1500]}" for log in useful_logs]
        )
        prompt = (
            "从工具观察中提炼最多6条证据。输出 JSON 数组，元素格式："
            '{"source":"来源URL或标识","claim":"证据支持的陈述","excerpt":"关键摘录","confidence":0.0-1.0,'
            '"supports":["候选答案或方向"],"contradicts":["候选答案或方向"],'
            '"evidence_type":"webpage_fact/image_fact/ocr/graph/kb_chunk/hypothesis/ambiguity",'
            '"premise_dependencies":["未直接观测到但推理依赖的前提"],'
            '"ambiguity":0.0-1.0,"exclusivity":0.0-1.0}。'
            "要求：最多6条，避免重复；优先提炼网页正文、图像事实、OCR、图谱证据、知识库片段；"
            "不要把报错文本当成证据；如果只是基于常识或代理文本的解释而非直接观测，请把 evidence_type 标成 hypothesis，"
            "并提高 ambiguity、写明 premise_dependencies。"
            "如果观察中存在相互冲突的解释，尽可能同时保留至少一条支持证据和一条反证证据，"
            "尽量不要只保留单边。"
            "若反证来自图像可见事实、OCR或高置信视觉 hypothesis，不要因为它与主流猜测不一致而省略。"
            "来自 fallback/proxy 的解释性文本只能算 hypothesis，且 ambiguity 不得低于 0.35。"
        )
        data = self.llm.chat_json(
            [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": f"主问题: {main_question}\n子问题: {sub_question}\n观察:\n{compact_logs}",
                },
            ],
            fallback=[],
        )

        evidence: list[Evidence] = []
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                claim = str(item.get("claim", "")).strip()
                if not claim:
                    continue
                evidence.append(
                    Evidence(
                        source=str(item.get("source", "unknown")),
                        claim=claim,
                        excerpt=str(item.get("excerpt", "")),
                        confidence=self._clamp01(item.get("confidence", 0.5), default=0.5),
                        supports=[str(x).strip() for x in item.get("supports", []) if str(x).strip()]
                        if isinstance(item.get("supports", []), list) else [],
                        contradicts=[str(x).strip() for x in item.get("contradicts", []) if str(x).strip()]
                        if isinstance(item.get("contradicts", []), list) else [],
                        evidence_type=str(item.get("evidence_type", "text")).strip() or "text",
                        premise_dependencies=[str(x).strip() for x in item.get("premise_dependencies", []) if str(x).strip()]
                        if isinstance(item.get("premise_dependencies", []), list) else [],
                        ambiguity=self._clamp01(item.get("ambiguity", 0.0), default=0.0),
                        exclusivity=self._clamp01(item.get("exclusivity", 0.0), default=0.0),
                    )
                )
        return evidence[:6]
    
    def _filter_docs_for_graph(self, question: str, docs: list[dict[str, str]], keep_k: int = 8) -> list[dict[str, str]]:
        q_terms = self._question_terms_for_retrieval(question)
        kept: list[tuple[float, dict[str, str]]] = []

        for d in docs:
            source = str(d.get("source", "")).strip()
            text = str(d.get("text", "")).strip()
            if not text:
                continue

            low_source = source.lower()
            low_text = text.lower()
            merged = f"{low_source}\n{low_text}"

            if low_source.endswith("#hypothesis") or low_source.endswith("#ambiguity"):
                continue
            if "图像候选解释" in text or "图像歧义" in text:
                continue

            if low_source.startswith(("local_image", "seed_local", "current_image", "vision_recheck", "reinspect:", "lexical:")):
                kept.append((10.0, {"source": source, "text": text}))
                continue

            overlap = sum(1 for t in q_terms if t in merged)

            if low_source == "rationale" or low_source.startswith("rationale") or low_source == "reasoning_chains":
                if overlap >= 2:
                    kept.append((2.0 + 0.5 * overlap, {"source": source, "text": text}))
                continue

            if text.startswith("搜索结果摘要:"):
                if overlap < 2:
                    continue
                score = 1.0 + overlap
            else:
                if overlap < 1:
                    continue
                score = float(overlap)

            kept.append((score, {"source": source, "text": text}))

        kept.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in kept[:keep_k]]
    
    def _should_run_tail_kg(
        self,
        main_question: str,
        sub_question: SubQuestion,
        logs: list[ToolObservation],
        docs: list[dict[str, str]],
        route_mode: str = "",
    ) -> bool:
        if not docs:
            return False

        if self._should_block_kg_for_visual_question(
            main_question=main_question,
            sub_question=sub_question,
            logs=logs,
        ):
            logs.append(
                ToolObservation(
                    tool_name="kg_gating",
                    input_data={
                        "main_question": main_question,
                        "sub_question": sub_question.question,
                        "route_mode": route_mode,
                    },
                    output_data=json.dumps(
                        {
                            "reason": "blocked_by_strong_visual_signal",
                            "tool": "tail_kg",
                        },
                        ensure_ascii=False,
                    ),
                )
            )
            return False

        for log in logs:
            if log.tool_name == "retrieve_graph_evidence" and not self._is_error_observation(log):
                return False

        normalized_route = str(route_mode or "").strip().lower()

        if normalized_route in {"", "auto", "direct_inference", "vision_recheck", "light_rag", "proxy_graph"}:
            logs.append(
                ToolObservation(
                    tool_name="kg_gating",
                    input_data={
                        "main_question": main_question,
                        "sub_question": sub_question.question,
                        "route_mode": normalized_route,
                    },
                    output_data=json.dumps(
                        {
                            "reason": "tail_kg_disabled_for_route",
                            "route_mode": normalized_route,
                        },
                        ensure_ascii=False,
                    ),
                )
            )
            return False

        allowed = normalized_route in {"full_rag", "agentic_rag", "kg", "hybrid"}
        if not allowed:
            logs.append(
                ToolObservation(
                    tool_name="kg_gating",
                    input_data={
                        "main_question": main_question,
                        "sub_question": sub_question.question,
                        "route_mode": normalized_route,
                    },
                    output_data=json.dumps(
                        {
                            "reason": "tail_kg_disabled_for_route",
                            "route_mode": normalized_route,
                        },
                        ensure_ascii=False,
                    ),
                )
            )
        return allowed
    

    def _rewrite_subquestion_from_visual_feedback(
        self,
        main_question: str,
        bad_sub_question: SubQuestion,
        logs: list[ToolObservation],
        failure_reason: str,
        used_questions: list[str],
    ) -> SubQuestion:
        visual_context = self._latest_visual_semantic_context(logs)
        prompt = (
            "请改写当前失败的子问题。"
            "要求新问题相比原有问题必须更贴近图像中可验证的视觉特征、OCR、局部区域或可直接观察到的关系；"
            "避免继续询问泛知识定义；"
            "不要重复已用问题；"
            '只输出 JSON 对象：{"question":"...","intent":"evidence","priority":1}'
        )
        data = self.llm.chat_json(
            [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        f"主问题: {main_question}\n"
                        f"失败子问题: {bad_sub_question.question}\n"
                        f"失败原因: {failure_reason}\n"
                        f"已用问题: {used_questions}\n"
                        f"视觉线索:\n{visual_context}"
                    ),
                },
            ],
            fallback={
                "question": bad_sub_question.question,
                "intent": bad_sub_question.intent,
                "priority": bad_sub_question.priority,
            },
            route="reasoning",
        )

        q = (
            str(data.get("question", bad_sub_question.question)).strip()
            if isinstance(data, dict) else bad_sub_question.question
        )

        if not q or q in used_questions:
            q = bad_sub_question.question

        return SubQuestion(
            question=q,
            intent=str(getattr(bad_sub_question, "intent", "evidence")),
            priority=int(getattr(bad_sub_question, "priority", 1)),
        )

    def _should_search_web_for_subquestion(
        self,
        sub_question: SubQuestion,
        logs: list[ToolObservation],
    ) -> bool:
        if not self._web_search_tool_enabled():
            return False

        if not self._has_question_image_input():
            return True

        visual_context = self._latest_visual_semantic_context(logs).strip()
        q = str(sub_question.question).lower()

        visual_like = any(k in q for k in [
            "text", "label", "sign", "logo", "marking", "lights", "shape", "region",
            "文字", "标识", "标签", "灯", "形状", "区域", "标记",
        ])
        if visual_like and not visual_context:
            return False

        return True
    
    
    @staticmethod
    def _normalize_action_key(tool_name: str, args: dict[str, Any]) -> str:
        compact: dict[str, Any] = {}
        for k, v in sorted((args or {}).items(), key=lambda x: x[0]):
            if k in {"_raw_image_ref", "image_ref", "image_url"}:
                compact[k] = str(v or "")[:120]
            elif isinstance(v, (str, int, float, bool)) or v is None:
                compact[k] = v
            else:
                compact[k] = str(v)[:160]
        return json.dumps({"tool": tool_name, "args": compact}, ensure_ascii=False, sort_keys=True)

    
    def _is_low_value_search_observation(self, question: str, obs: ToolObservation) -> bool:
        if obs.tool_name != "search_web":
            return False
        result_count = self._search_results_count(obs)
        match_score = self._score_search_observation_match(question, obs)
        return result_count <= 0 or match_score < float(getattr(settings, "web_search_match_threshold", 0.55) or 0.55)
    
    
    @staticmethod
    def _configured_max_actions_per_question(default: int = 4) -> int:
        """
        MAX_ACTIONS_PER_QUESTION 是单个 sub-question 内允许的工具 action 硬上限。
        """
        try:
            value = int(getattr(settings, "max_actions_per_question", default) or default)
        except Exception:
            value = default
        return max(1, value)
    

    def investigate(
        self,
        main_question: str,
        sub_question: SubQuestion,
        seed_docs: list[dict[str, str]] | None = None,
        route_mode: str = "",
    ) -> tuple[list[Evidence], list[ToolObservation]]:
        logs: list[ToolObservation] = []
        normalized_route = str(route_mode or "").strip().lower()

        max_actions = self._configured_max_actions_per_question(default=4)
        actions_used = 0

        executed_actions: set[str] = set()
        low_value_search_rounds = 0
        low_value_graph_rounds = 0

        for _ in range(max_actions):
            if actions_used >= max_actions:
                break
            
            action = self._next_action(
                main_question=main_question,
                sub_question=sub_question,
                logs=logs,
            )
            tool_name = str(action.get("tool", "finish")).strip()
            args = action.get("args", {}) if isinstance(action.get("args"), dict) else {}

            if tool_name == "finish":
                break

            if tool_name == "retrieve_graph_evidence" and normalized_route and "route_mode" not in args:
                args["route_mode"] = normalized_route

            action_key = self._normalize_action_key(tool_name, args)
            if action_key in executed_actions:
                logs.append(
                    ToolObservation(
                        tool_name=tool_name,
                        input_data=args,
                        output_data="SKIPPED_DUPLICATE_ACTION",
                    )
                )
                continue
            executed_actions.add(action_key)

            if (not self._has_question_image_input()) and tool_name in self._image_tool_names():
                logs.append(
                    ToolObservation(
                        tool_name=tool_name,
                        input_data=args,
                        output_data="SKIPPED_NO_IMAGE_INPUT",
                    )
                )
                actions_used += 1
                continue

            try:
                obs = self.tool_router.run(tool_name=tool_name, args=args)
            except Exception as exc:
                obs = ToolObservation(
                    tool_name=tool_name,
                    input_data=args,
                    output_data=f"TOOL_ERROR: {exc}",
                )
            logs.append(obs)
            actions_used += 1

            if self._web_search_tool_enabled() and tool_name == "search_web":
                if self._is_low_value_search_observation(sub_question.question, obs):
                    low_value_search_rounds += 1
                else:
                    low_value_search_rounds = 0

                if low_value_search_rounds < 2 and actions_used < max_actions:
                    top_k = 5
                    try:
                        top_k = min(max(1, int(args.get("top_k", 5))), 6)
                    except Exception:
                        top_k = 5

                    retry_logs = self._retry_search_with_variants(
                        main_question=main_question,
                        sub_question=sub_question,
                        logs=logs,
                        previous_obs=obs,
                        top_k=top_k,
                        remaining_budget=max_actions - actions_used,
                    )
                    logs.extend(retry_logs)
                    actions_used += len(retry_logs)

                    if actions_used < max_actions:
                        open_urls = self._should_open_webpage_from_search(
                            sub_question=sub_question,
                            obs=obs,
                        )
                    else:
                        open_urls = []

                    for open_url in open_urls[:2]:
                        if actions_used >= max_actions:
                            break

                        open_key = self._normalize_action_key("open_webpage", {"url": open_url})
                        if open_key in executed_actions:
                            continue
                        executed_actions.add(open_key)

                        try:
                            page_obs = self.tool_router.run(
                                tool_name="open_webpage",
                                args={"url": open_url},
                            )
                        except Exception as exc:
                            page_obs = ToolObservation(
                                tool_name="open_webpage",
                                input_data={"url": open_url},
                                output_data=f"TOOL_ERROR: {exc}",
                            )
                        logs.append(page_obs)
                        actions_used += 1

                if low_value_search_rounds >= 2:
                    logs.append(
                        ToolObservation(
                            tool_name="search_web",
                            input_data={"question": sub_question.question},
                            output_data="STOPPED_LOW_VALUE_SEARCH",
                        )
                    )

            if tool_name == "retrieve_graph_evidence":
                low_graph = False
                try:
                    parsed = self._safe_json_loads(str(obs.output_data or ""), fallback={})
                    cov = float(parsed.get("coverage", 0.0) or 0.0) if isinstance(parsed, dict) else 0.0
                    link = parsed.get("linking_artifacts", {}) if isinstance(parsed, dict) else {}
                    selected = str(link.get("selected_strategy", "")).strip().lower() if isinstance(link, dict) else ""
                    low_graph = (cov <= 0.0) or ("unknown_backend" in selected) or ("no_backend_available" in selected)
                except Exception:
                    low_graph = True

                low_value_graph_rounds = low_value_graph_rounds + 1 if low_graph else 0
                if low_value_graph_rounds >= 2:
                    logs.append(
                        ToolObservation(
                            tool_name="retrieve_graph_evidence",
                            input_data={"question": sub_question.question},
                            output_data="STOPPED_LOW_VALUE_GRAPH_RETRIEVAL",
                        )
                    )

            if tool_name == "extract_images":
                logs.extend(self._auto_consume_image_candidates(sub_question=sub_question, logs=logs))

            if tool_name == "retrieve_graph_evidence":
                logs.extend(self._auto_consume_kb_image_candidates(sub_question=sub_question, logs=logs))

        docs = list(seed_docs or [])
        docs.extend(self._logs_to_documents(logs))
        docs = self._rerank_docs_for_retrieval(
            question=sub_question.question,
            docs=docs,
            keep_k=int(getattr(settings, "embedding_rerank_keep_k", 6) or 6),
        )

        # route 允许时，末尾再补一轮 tail KG，避免主循环里没有稳定触发 KG
        should_tail_kg = False
        if hasattr(self, "_should_run_tail_kg"):
            try:
                should_tail_kg = bool(
                    self._should_run_tail_kg(
                        main_question=main_question,
                        sub_question=sub_question,
                        logs=logs,
                        docs=docs,
                        route_mode=normalized_route,
                    )
                )
            except Exception:
                should_tail_kg = False

        if should_tail_kg:
            tail_args = {
                "question": sub_question.question,
                "schema": self._proxy_schema_for_question(main_question),
                "graph_context": "\n".join([f"[{d['source']}] {d['text'][:250]}" for d in docs[:6]]),
                "documents": docs[:8],
                "route_mode": normalized_route,
            }

            if not self._has_question_image_input():
                tail_args["allow_kb_augmentation"] = False

            tail_key = self._normalize_action_key("retrieve_graph_evidence", tail_args)
            if tail_key not in executed_actions and low_value_graph_rounds < 2:
                executed_actions.add(tail_key)
                try:
                    tail_obs = self.tool_router.run(
                        tool_name="retrieve_graph_evidence",
                        args=tail_args,
                    )
                except Exception as exc:
                    tail_obs = ToolObservation(
                        tool_name="retrieve_graph_evidence",
                        input_data=tail_args,
                        output_data=f"TOOL_ERROR: {exc}",
                    )
                logs.append(tail_obs)
                logs.extend(self._auto_consume_kb_image_candidates(sub_question=sub_question, logs=logs))

                docs = list(seed_docs or [])
                docs.extend(self._logs_to_documents(logs))
                docs = self._rerank_docs_for_retrieval(
                    question=sub_question.question,
                    docs=docs,
                    keep_k=int(getattr(settings, "embedding_rerank_keep_k", 6) or 6),
                )

        # 先走确定性 lifting，再叠加 LLM evidence 蒸馏
        deterministic_evidence: list[Evidence] = []
        if hasattr(self, "_lift_documents_to_evidence"):
            try:
                deterministic_evidence = list(self._lift_documents_to_evidence(docs))
            except Exception:
                deterministic_evidence = []

        try:
            llm_evidence = self._distill_evidence(
                main_question=main_question,
                sub_question=sub_question.question,
                logs=logs,
            )
        except Exception as exc:
            llm_evidence = []
            logs.append(
                ToolObservation(
                    tool_name="distill_evidence",
                    input_data={
                        "main_question": main_question,
                        "sub_question": sub_question.question,
                    },
                    output_data=f"TOOL_ERROR: {type(exc).__name__}: {exc}",
                )
            )

        evidence = deterministic_evidence + llm_evidence
        if hasattr(self, "_dedupe_evidence"):
            try:
                evidence = self._dedupe_evidence(evidence, limit=8)
            except Exception:
                pass

        # 显式写出 grounding check，供 workflow.py / eval_vqa.py 消费
        try:
            grounding_accept, grounding_score, grounding_reason = self._assess_subquestion_visual_grounding(
                main_question=main_question,
                sub_question=sub_question,
                evidence=evidence,
                logs=logs,
            )
        except Exception as exc:
            grounding_accept = True
            grounding_score = 1.0
            grounding_reason = f"grounding_check_error:{type(exc).__name__}"

        logs.append(
            ToolObservation(
                tool_name="subquestion_grounding_check",
                input_data={
                    "main_question": main_question,
                    "sub_question": sub_question.question,
                    "intent": getattr(sub_question, "intent", ""),
                    "priority": getattr(sub_question, "priority", 0),
                },
                output_data=json.dumps(
                    {
                        "accept": bool(grounding_accept),
                        "score": float(self._clamp01(grounding_score, default=0.0)),
                        "reason": str(grounding_reason or ""),
                    },
                    ensure_ascii=False,
                ),
            )
        )

        return evidence, logs


class CriticAgent:
    def __init__(self, llm: QwenVLClient) -> None:
        self.llm = llm

    def _clamp01(self, value: Any, default: float = 0.0) -> float:
        try:
            x = float(value)
        except Exception:
            return default
        return max(0.0, min(1.0, x))

    def critique(self, question: str, evidence: list[Evidence]) -> CritiqueResult:
        evidence_brief = "\n\n".join(
            [
                (
                    f"来源: {e.source}\n"
                    f"结论: {e.claim}\n"
                    f"摘录: {e.excerpt[:300]}\n"
                    f"置信度: {e.confidence}\n"
                    f"类型: {e.evidence_type}\n"
                    f"支持: {e.supports}\n"
                    f"反驳: {e.contradicts}\n"
                    f"前提依赖: {e.premise_dependencies}\n"
                    f"歧义度: {e.ambiguity}\n"
                    f"排他性: {e.exclusivity}"
                )
                for e in evidence[:12]
            ]
        )
        prompt = (
            '判断当前证据是否足够回答主问题。输出 JSON 对象：'
            '{"sufficient":true/false,'
            '"follow_up_questions":[...],'
            '"missing_dimensions":[...],'
            '"must_reinspect_image":true/false,'
            '"must_abort_finalization":true/false,'
            '"dominant_failure_mode":"visual_ambiguity/unsupported_assumption/low_exclusivity/insufficient_evidence/...",'
            '"final_confidence":0.0-1.0,'
            '"has_unresolved_contradiction":true/false}。'
            "其中 final_confidence 表示对“最终必须作答”这一行为的总体置信度，而不是单条证据分数。"
            "规则："
            "如果存在高置信图像/OCR/视觉 hypothesis 反证，而当前证据并未明确解释为何仍应选择某个答案，"
            "则 has_unresolved_contradiction 必须为 true。"
            "当 has_unresolved_contradiction 为 true 时，若 final_confidence < 0.75，应优先将 must_reinspect_image 设为 true，"
            "且 must_abort_finalization 设为 true。"
            "若 final_confidence >= 0.60 且不存在未解决冲突，则必须继续给出一个最确定的答案，此时 must_abort_finalization 必须为 false；"
            "若 final_confidence < 0.60，则允许弃权。"
        )
        
        data = self.llm.chat_json(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"主问题: {question}\n\n证据摘要:\n{evidence_brief}"},
            ],
            fallback={
                "sufficient": False,
                "follow_up_questions": [question],
                "missing_dimensions": ["fallback"],
                "must_reinspect_image": False,
                "must_abort_finalization": True,
                "dominant_failure_mode": "fallback",
                "final_confidence": 0.0,
                "has_unresolved_contradiction": False,
            },
            route="verify",
        )

        if not isinstance(data, dict):
            data = {
                "sufficient": False,
                "follow_up_questions": [question],
                "missing_dimensions": ["fallback"],
                "must_reinspect_image": False,
                "must_abort_finalization": True,
                "dominant_failure_mode": "fallback",
                "final_confidence": 0.0,
                "has_unresolved_contradiction": False,
            }

        final_confidence = self._clamp01(data.get("final_confidence", 0.0), default=0.0)
        unresolved = bool(data.get("has_unresolved_contradiction", False))
        must_reinspect_image = bool(data.get("must_reinspect_image", False))
        must_abort_finalization = bool(data.get("must_abort_finalization", False))
        sufficient = bool(data.get("sufficient", False))

        if unresolved and final_confidence < 0.75:
            must_reinspect_image = True
            must_abort_finalization = True
        elif final_confidence >= 0.60 and not unresolved:
            must_abort_finalization = False
        else:
            must_abort_finalization = must_abort_finalization or (not sufficient)

        return CritiqueResult(
            sufficient=sufficient,
            follow_up_questions=[str(x) for x in data.get("follow_up_questions", [])][:3]
            if isinstance(data.get("follow_up_questions", []), list) else [],
            missing_dimensions=[str(x) for x in data.get("missing_dimensions", [])][:4]
            if isinstance(data.get("missing_dimensions", []), list) else [],
            must_reinspect_image=must_reinspect_image,
            must_abort_finalization=must_abort_finalization,
            dominant_failure_mode=str(data.get("dominant_failure_mode", "")).strip(),
            final_confidence=final_confidence,
            has_unresolved_contradiction=unresolved,
        )


class VerifierAgent:
    def __init__(self, llm: QwenVLClient) -> None:
        self.llm = llm

    def _is_visual_evidence(self, e: Evidence) -> bool:
        source = str(e.source or "").lower()
        return (
            e.evidence_type in {"image_fact", "ocr", "hypothesis"}
            or "local_image" in source
            or "vision_recheck" in source
            or "image" in source
        )

    def _is_contradiction_evidence(self, e: Evidence) -> bool:
        if isinstance(e.contradicts, list) and any(str(x).strip() for x in e.contradicts):
            return True

        merged = f"{e.claim} {e.excerpt}".lower()
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
        return any(m in merged for m in markers)

    def _is_support_evidence(self, e: Evidence) -> bool:
        if isinstance(e.supports, list) and any(str(x).strip() for x in e.supports):
            return True

        if self._is_contradiction_evidence(e):
            return False

        return (
            e.confidence >= 0.65
            and e.ambiguity <= 0.45
            and e.evidence_type in {"image_fact", "ocr", "graph", "kb_chunk", "webpage_fact", "text"}
        )

    def _rank_score(self, e: Evidence) -> float:
        score = (
            1.40 * float(e.confidence)
            + 0.55 * float(e.exclusivity)
            - 0.90 * float(e.ambiguity)
            - 0.12 * len(e.premise_dependencies)
        )

        type_bonus = {
            "image_fact": 0.22,
            "ocr": 0.20,
            "graph": 0.16,
            "kb_chunk": 0.14,
            "webpage_fact": 0.12,
            "text": 0.08,
            "ambiguity": -0.10,
            "hypothesis": -0.02,
        }
        score += type_bonus.get(e.evidence_type, 0.0)

        if self._is_visual_evidence(e):
            score += 0.12
        if self._is_contradiction_evidence(e):
            score += 0.18

        # 仅当 hypothesis 不是来自真实视觉链时，才明显降权
        if e.evidence_type == "hypothesis" and not self._is_visual_evidence(e):
            score -= 0.20

        return score

    def _fallback_rank(self, evidence: list[Evidence]) -> list[Evidence]:
        ranked = sorted(evidence, key=self._rank_score, reverse=True)
        return ranked[:8]

    def _rebalance_verified_evidence(
        self,
        kept: list[Evidence],
        all_evidence: list[Evidence],
    ) -> list[Evidence]:
        ranked_pool = self._fallback_rank(all_evidence if all_evidence else kept)

        selected: list[Evidence] = []
        seen: set[tuple[str, str, str]] = set()

        def _push(e: Evidence | None) -> None:
            if e is None:
                return
            key = (str(e.source), str(e.claim), str(e.excerpt))
            if key in seen:
                return
            seen.add(key)
            selected.append(e)

        best_visual = next((e for e in ranked_pool if self._is_visual_evidence(e)), None)
        best_contradiction = next((e for e in ranked_pool if self._is_contradiction_evidence(e)), None)
        best_support = next((e for e in ranked_pool if self._is_support_evidence(e)), None)

        for e in [best_visual, best_contradiction, best_support]:
            _push(e)

        for e in kept:
            _push(e)

        for e in ranked_pool:
            if len(selected) >= 6:
                break
            _push(e)

        return selected[:6]

    def verify(self, question: str, evidence: list[Evidence]) -> list[Evidence]:
        if not evidence:
            return []

        packed = "\n\n".join(
            [
                (
                    f"[{idx}] 来源={e.source}\n"
                    f"陈述={e.claim}\n"
                    f"摘录={e.excerpt[:250]}\n"
                    f"置信度={e.confidence}\n"
                    f"类型={e.evidence_type}\n"
                    f"支持={e.supports}\n"
                    f"反驳={e.contradicts}\n"
                    f"前提依赖={e.premise_dependencies}\n"
                    f"歧义度={e.ambiguity}\n"
                    f"排他性={e.exclusivity}"
                )
                for idx, e in enumerate(evidence)
            ]
        )
        prompt = (
            "对每条证据进行有效性筛选。返回 JSON 数组，元素为应保留的索引（整数）。"
            "标准：与主问题直接相关、有来源、有可验证摘录、尽量不严重依赖未直接观测到的前提、"
            "对候选答案具有区分能力。优先保留 image_fact / ocr / graph / kb_chunk / webpage_fact，谨慎保留 hypothesis。"
            "若存在直接冲突证据，必须同时至少保留一条支持证据与一条反证证据。"
            "最多保留 6 条证据。"
        )
        
        data = self.llm.chat_json(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"主问题: {question}\n\n证据候选:\n{packed}"},
            ],
            fallback=[],
            route="verify",
        )

        keep: list[Evidence] = []
        if isinstance(data, list):
            for idx in data:
                if isinstance(idx, int) and 0 <= idx < len(evidence):
                    keep.append(evidence[idx])

        if keep:
            return self._rebalance_verified_evidence(keep[:6], evidence)

        return self._rebalance_verified_evidence(self._fallback_rank(evidence), evidence)

class SynthesizerAgent:
    def __init__(self, llm: QwenVLClient) -> None:
        self.llm = llm

    def _clamp01(self, value: Any, default: float = 0.0) -> float:
        try:
            x = float(value)
        except Exception:
            return default
        return max(0.0, min(1.0, x))

    def _merged_text(self, e: Evidence) -> str:
        return f"{e.claim} {e.excerpt}".strip().lower()

    
    def _extract_choices_from_question(self, question: str) -> list[str]:
        text = str(question or "")
        out: list[str] = []

        for m in re.finditer(r"(?:^|\n)\s*([A-H])\.\s*(.+)", text):
            choice_text = str(m.group(2)).strip()
            if choice_text:
                out.append(choice_text)

        return out

    def _norm_choice_text(self, text: str) -> str:
        s = str(text or "").strip().lower()
        s = re.sub(r"^[\(\[]?[a-zA-Z][\)\].:\-]\s*", "", s)
        s = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", s)
        return s

    def _map_answer_to_choices(self, answer: str, choices: list[str]) -> str:
        raw = str(answer or "").strip()
        if not raw or not choices:
            return raw

        upper = raw.upper().strip()
        if len(upper) == 1 and "A" <= upper <= "H":
            idx = ord(upper) - ord("A")
            if 0 <= idx < len(choices):
                return choices[idx]

        raw_norm = self._norm_choice_text(raw)
        if not raw_norm:
            return raw

        best_choice = None
        best_score = -1.0

        for ch in choices:
            ch_text = str(ch).strip()
            ch_norm = self._norm_choice_text(ch_text)
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

        return raw
    
    
    def _nonempty_str_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        out: list[str] = []
        for x in value:
            s = str(x).strip()
            if s:
                out.append(s)
        return out

    def _is_contradiction_evidence(self, e: Evidence) -> bool:
        if self._nonempty_str_list(e.contradicts):
            return True

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
        text = f" {self._merged_text(e)} "
        return any(m in text for m in markers)

    def _is_support_evidence(self, e: Evidence) -> bool:
        if self._nonempty_str_list(e.supports):
            return True
        if self._is_contradiction_evidence(e):
            return False

        return (
            e.confidence >= 0.65
            and e.ambiguity <= 0.45
            and e.evidence_type in {"image_fact", "ocr", "graph", "kb_chunk", "webpage_fact", "text"}
        )

    def _evidence_score(self, e: Evidence) -> float:
        score = (
            1.35 * float(e.confidence)
            + 0.55 * float(e.exclusivity)
            - 0.85 * float(e.ambiguity)
            - 0.10 * len(self._nonempty_str_list(e.premise_dependencies))
        )

        if e.evidence_type in {"image_fact", "ocr"}:
            score += 0.15

        if self._is_contradiction_evidence(e):
            score += 0.12

        return score

    def _pick_answer_from_evidence(self, e: Evidence, question: str = "") -> str:
        choices = self._extract_choices_from_question(question)

        supports = self._nonempty_str_list(e.supports)
        if supports:
            ans = supports[0]
            return self._map_answer_to_choices(ans, choices) if choices else ans

        claim = str(e.claim).strip()
        if claim:
            ans = claim[:120]
            return self._map_answer_to_choices(ans, choices) if choices else ans

        excerpt = str(e.excerpt).strip()
        if excerpt:
            ans = excerpt[:120]
            return self._map_answer_to_choices(ans, choices) if choices else ans

        return ""

    def _top_evidence(self, evidence: list[Evidence], limit: int = 12) -> list[Evidence]:
        ranked = sorted(evidence, key=self._evidence_score, reverse=True)
        return ranked[:limit]

    def _format_evidence_block(self, e: Evidence, idx: int) -> str:
        supports = self._nonempty_str_list(e.supports)
        contradicts = self._nonempty_str_list(e.contradicts)
        premise_dependencies = self._nonempty_str_list(e.premise_dependencies)

        return (
            f"[证据{idx}]\n"
            f"来源: {e.source}\n"
            f"陈述: {str(e.claim).strip()[:200]}\n"
            f"摘录: {str(e.excerpt).strip()[:400]}\n"
            f"置信度: {self._clamp01(e.confidence, 0.0):.3f}\n"
            f"类型: {e.evidence_type}\n"
            f"支持: {supports}\n"
            f"反驳: {contradicts}\n"
            f"前提依赖: {premise_dependencies}\n"
            f"歧义度: {self._clamp01(e.ambiguity, 0.0):.3f}\n"
            f"排他性: {self._clamp01(e.exclusivity, 0.0):.3f}"
        )

    def _build_evidence_text(self, evidence: list[Evidence], limit: int = 12) -> str:
        top_items = self._top_evidence(evidence, limit=limit)
        if not top_items:
            return "无有效证据"

        return "\n\n".join(
            [self._format_evidence_block(e, idx + 1) for idx, e in enumerate(top_items)]
        )


    def _fallback_decision(self, evidence: list[Evidence], question: str = "") -> FinalAnswerDecision:
        if not evidence:
            return FinalAnswerDecision(
                answer="ABSTAIN",
                confidence=0.0,
                abstain=True,
                reason="no_evidence",
            )

        ranked = self._top_evidence(evidence, limit=len(evidence))
        support_bucket = [e for e in ranked if self._is_support_evidence(e)]
        contradiction_bucket = [e for e in ranked if self._is_contradiction_evidence(e)]

        best_support = support_bucket[0] if support_bucket else None
        best_contradiction = contradiction_bucket[0] if contradiction_bucket else None

        if best_support and best_contradiction:
            support_score = self._evidence_score(best_support)
            contradiction_score = self._evidence_score(best_contradiction)

            if support_score >= contradiction_score + 0.18:
                answer = self._pick_answer_from_evidence(best_support, question=question)
                confidence = self._clamp01(
                    0.72 * float(best_support.confidence)
                    + 0.18 * float(best_support.exclusivity)
                    + 0.10 * (1.0 - float(best_support.ambiguity))
                    - 0.10,
                    default=0.0,
                )
                abstain = confidence < 0.60 or not answer
                if abstain:
                    answer = "ABSTAIN"

                return FinalAnswerDecision(
                    answer=answer or "ABSTAIN",
                    confidence=confidence,
                    abstain=abstain,
                    reason="fallback_support_over_contradiction",
                )

            return FinalAnswerDecision(
                answer="ABSTAIN",
                confidence=min(
                    0.58,
                    self._clamp01(
                        max(float(best_support.confidence), float(best_contradiction.confidence)),
                        default=0.55,
                    ),
                ),
                abstain=True,
                reason="unresolved_conflict_fallback",
            )

        best = ranked[0]
        confidence = self._clamp01(
            0.70 * float(best.confidence)
            + 0.20 * float(best.exclusivity)
            + 0.10 * (1.0 - float(best.ambiguity))
            - 0.05 * len(self._nonempty_str_list(best.premise_dependencies)),
            default=0.0,
        )

        answer = self._pick_answer_from_evidence(best, question=question)
        abstain = confidence < 0.60 or not answer
        if abstain:
            answer = "ABSTAIN"

        return FinalAnswerDecision(
            answer=answer or "ABSTAIN",
            confidence=confidence,
            abstain=abstain,
            reason="fallback_from_ranked_evidence",
        )

    def decide(self, state: ResearchState) -> FinalAnswerDecision:
        fallback = self._fallback_decision(state.evidence, question=state.question)
        evidence_text = self._build_evidence_text(state.evidence, limit=12)

        prompt = (
            "假设你是一个专业的最终答案决策 Agent。"
            "请基于证据直接做最终决策。"
            '仅输出 JSON 对象：{"answer":"...","confidence":0.0-1.0,"abstain":true/false,"reason":"..."}。'
            "规则："
            "1) confidence 表示你对最终答案正确性的总体置信度；"
            "2) 若 confidence >= 0.60，则必须给出一个最确定的答案，abstain 必须为 false；"
            "3) 若 confidence < 0.60，则允许 abstain=true；"
            "4) 如果主问题中已经包含选项，请尽量输出最接近标准选项的短答案；"
            "5) 不要输出解释性长文，不要输出 markdown。"
        )

        data = self.llm.chat_json(
            [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        f"主问题: {state.question}\n\n"
                        f"证据库:\n{evidence_text}\n\n"
                        f"回退决策参考:\n"
                        f"answer={fallback.answer}, confidence={fallback.confidence}, "
                        f"abstain={fallback.abstain}, reason={fallback.reason}"
                    ),
                },
            ],
            fallback={
                "answer": fallback.answer,
                "confidence": fallback.confidence,
                "abstain": fallback.abstain,
                "reason": fallback.reason,
            },
            route="synthesize",
        )

        if not isinstance(data, dict):
            return fallback

        choices = self._extract_choices_from_question(state.question)
        answer = str(data.get("answer", "")).strip()
        if choices:
            answer = self._map_answer_to_choices(answer, choices)
        confidence = self._clamp01(data.get("confidence", fallback.confidence), default=fallback.confidence)
        abstain = bool(data.get("abstain", fallback.abstain))
        reason = str(data.get("reason", "")).strip() or fallback.reason

        if confidence >= 0.60:
            abstain = False
            if not answer or answer.upper() == "ABSTAIN":
                answer = fallback.answer
                if not answer or answer.upper() == "ABSTAIN":
                    answer = "UNKNOWN"
        else:
            if abstain or not answer:
                abstain = True
                answer = "ABSTAIN"

        return FinalAnswerDecision(
            answer=answer,
            confidence=confidence,
            abstain=abstain,
            reason=reason,
        )

    def synthesize(self, state: ResearchState) -> str:
        evidence_text = self._build_evidence_text(state.evidence, limit=12)

        prompt = (
            "假设你是一个专业的精简证据摘要 Agent。"
            "请输出结构化中文报告：\n"
            "1) 执行摘要\n"
            "2) 分析框架\n"
            "3) 关键证据（按来源编号）\n"
            "4) 结论与适用边界\n"
            "5) 风险与不确定性\n"
            "6) 后续研究建议\n"
            "要求："
            "必须引用提供的证据，不得凭空编造；"
            "如果证据中存在高歧义、低排他性或强前提依赖，请在风险与不确定性中明确指出；"
            "保持简洁，不要写无关铺垫。"
        )

        return self.llm.chat(
            [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": f"主问题: {state.question}\n\n证据库:\n{evidence_text}",
                },
            ],
            temperature=0.1,
            route="synthesize",
        )
        
        
