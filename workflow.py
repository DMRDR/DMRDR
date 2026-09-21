from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from tools import ToolObservation, parse_loose_json_object

import json
import re
from collections import defaultdict
from agents import (
    CriticAgent,
    Evidence,
    FinalAnswerDecision,
    PlannerAgent,
    ResearchState,
    ResearcherAgent,
    SubQuestion,
    SynthesizerAgent,
    VerifierAgent,
)
from config import settings


@dataclass
class WorkflowResult:
    report: str
    state: ResearchState
    decision: FinalAnswerDecision | None = None
    final_critique: Any | None = None
    critique_trace: list[dict[str, Any]] = field(default_factory=list)
    route_trace: dict[str, Any] = field(default_factory=dict)
    error_trace: list[dict[str, Any]] = field(default_factory=list)
    decision_status: str = "finalized"
    final_confidence: float = 0.0


class DeepResearchWorkflow:
    """
    plan -> investigate(with tool actions) -> critique -> iterate -> verify -> synthesize
    """

    def __init__(
        self,
        planner: PlannerAgent,
        researcher: ResearcherAgent,
        critic: CriticAgent,
        verifier: VerifierAgent,
        synthesizer: SynthesizerAgent,
    ) -> None:
        self.planner = planner
        self.researcher = researcher
        self.critic = critic
        self.verifier = verifier
        self.synthesizer = synthesizer

    @staticmethod
    def _score_evidence_importance(question: str, evidence: Evidence) -> float:
        q_terms = {t for t in question.lower().split() if t}
        text = f"{evidence.claim} {evidence.excerpt}".lower()
        overlap = sum(1 for t in q_terms if t in text)
        overlap_score = overlap / max(len(q_terms), 1)

        ambiguity = float(getattr(evidence, "ambiguity", 0.0))
        exclusivity = float(getattr(evidence, "exclusivity", 0.0))
        premise_penalty = 1.0 if getattr(evidence, "premise_dependencies", []) else 0.0

        score = (
            0.45 * float(evidence.confidence)
            + 0.20 * overlap_score
            + 0.20 * exclusivity
            - 0.10 * ambiguity
            - 0.05 * premise_penalty
        )
        return score
    
    
    @staticmethod
    def _clamp01(value: Any, default: float = 0.0) -> float:
        try:
            x = float(value)
        except Exception:
            return default
        return max(0.0, min(1.0, x))

    
    @staticmethod
    def _mean_conf(values: list[float], default: float = 0.0) -> float:
        vals = [float(v) for v in values if v is not None]
        if not vals:
            return default
        return sum(vals) / len(vals)

    
    def _resolve_auto_route(
        self,
        question: str,
        image_ref: str | None = None,
        seed_docs: list[dict[str, str]] | None = None,
    ) -> str:
        q = str(question or "").lower()
        has_image = bool(str(image_ref or "").strip())
        has_docs = bool(seed_docs)

        knowledge_markers = [
            "why", "how", "who", "which", "compare", "difference", "because",
            "为什么", "如何", "谁", "哪个", "比较", "差异", "原因",
        ]
        visual_markers = [
            "image", "picture", "photo", "color", "shape", "text", "ocr", "region",
            "图片", "图像", "颜色", "形状", "文字", "区域",
        ]

        if has_image and any(k in q for k in visual_markers) and not has_docs:
            return "light_rag"

        if has_docs and any(k in q for k in knowledge_markers):
            return "full_rag"

        if has_image and has_docs:
            return "full_rag"
        if has_docs:
            return "light_rag"
        if has_image:
            return "light_rag"

        return "direct_inference"

    def _merge_seed_docs(
        self,
        base_docs: list[dict[str, str]] | None,
        tool_logs: list[ToolObservation] | None,
    ) -> list[dict[str, str]]:
        merged: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()

        for d in list(base_docs or []) + self.researcher._logs_to_documents(list(tool_logs or [])):
            if not isinstance(d, dict):
                continue
            source = str(d.get("source", "")).strip()
            text = str(d.get("text", "")).strip()
            if not text:
                continue
            key = (source, text[:300])
            if key in seen:
                continue
            seen.add(key)
            merged.append({"source": source, "text": text})

        return merged
    
    
    def _aggregate_visible_fact_confidence(self, visible_facts: dict[str, Any]) -> float:
        if not isinstance(visible_facts, dict) or not visible_facts:
            return 0.0

        non_empty = 0
        relation_like = 0

        for k, v in visible_facts.items():
            if v in {"", None, [], {}}:
                continue
            non_empty += 1

            key = str(k).lower()
            if any(tok in key for tok in ["holding", "wearing", "next_to", "on_", "under", "behind", "in_", "position", "relation"]):
                relation_like += 1

        score = 0.58 + 0.04 * min(non_empty, 5) + 0.03 * min(relation_like, 3)
        return self._clamp01(score, default=0.72)

    
    def _aggregate_ocr_confidence(self, ocr_items: list[dict[str, Any]]) -> float:
        if not isinstance(ocr_items, list) or not ocr_items:
            return 0.0

        vals: list[float] = []
        for item in ocr_items:
            if not isinstance(item, dict):
                continue
            vals.append(self._clamp01(item.get("confidence", 0.72), default=0.72))

        if not vals:
            return 0.72

        return self._clamp01(self._mean_conf(vals, default=0.72), default=0.72)
    

    def _compress_state_memory(self, state: ResearchState) -> None:
        if len(state.evidence) > settings.max_evidence_items:
            protected_types = {"image_fact", "ocr", "hypothesis", "ambiguity", "graph", "kb_chunk"}
            protected: list[Evidence] = []
            seen_types: set[str] = set()

            for e in reversed(state.evidence):
                et = str(getattr(e, "evidence_type", "")).strip().lower()
                if et in protected_types and et not in seen_types:
                    protected.append(e)
                    seen_types.add(et)

            protected.reverse()
            protected_ids = {id(e) for e in protected}

            remaining = [e for e in state.evidence if id(e) not in protected_ids]
            ranked = sorted(
                remaining,
                key=lambda e: self._score_evidence_importance(state.question, e),
                reverse=True,
            )

            keep_extra = max(0, settings.max_evidence_items - len(protected))
            state.evidence = protected + ranked[:keep_extra]

        if len(state.tool_logs) > settings.max_tool_logs:
            state.tool_logs = state.tool_logs[-settings.max_tool_logs:]

    def _safe_json_loads(self, text: str) -> dict:
        parsed = parse_loose_json_object(text, fallback={})
        return parsed if isinstance(parsed, dict) else {}
    
    def _append_tool_error(
        self,
        state: ResearchState,
        tool_name: str,
        input_data: dict[str, Any],
        exc: Exception,
    ) -> None:
        state.tool_logs.append(
            ToolObservation(
                tool_name=tool_name,
                input_data=input_data,
                output_data=f"TOOL_ERROR: {type(exc).__name__}: {exc}",
            )
        )

    @staticmethod
    def _evidence_to_dict(e: Evidence) -> dict[str, Any]:
        return {
            "source": str(getattr(e, "source", "")),
            "claim": str(getattr(e, "claim", "")),
            "excerpt": str(getattr(e, "excerpt", "")),
            "confidence": float(getattr(e, "confidence", 0.0) or 0.0),
            "supports": list(getattr(e, "supports", []) or []),
            "contradicts": list(getattr(e, "contradicts", []) or []),
            "evidence_type": str(getattr(e, "evidence_type", "")),
            "premise_dependencies": list(getattr(e, "premise_dependencies", []) or []),
            "ambiguity": float(getattr(e, "ambiguity", 0.0) or 0.0),
            "exclusivity": float(getattr(e, "exclusivity", 0.0) or 0.0),
        }

    @staticmethod
    def _tool_log_to_dict(log: ToolObservation) -> dict[str, Any]:
        return {
            "tool_name": str(getattr(log, "tool_name", "")),
            "input_data": getattr(log, "input_data", {}) or {},
            "output_data": str(getattr(log, "output_data", "")),
        }

    @staticmethod
    def _critique_to_dict(critique: Any) -> dict[str, Any]:
        if critique is None:
            return {}
        return {
            "sufficient": bool(getattr(critique, "sufficient", False)),
            "follow_up_questions": list(getattr(critique, "follow_up_questions", []) or []),
            "missing_dimensions": list(getattr(critique, "missing_dimensions", []) or []),
            "must_reinspect_image": bool(getattr(critique, "must_reinspect_image", False)),
            "must_abort_finalization": bool(getattr(critique, "must_abort_finalization", False)),
            "dominant_failure_mode": str(getattr(critique, "dominant_failure_mode", "")),
            "final_confidence": float(getattr(critique, "final_confidence", 0.0) or 0.0),
            "has_unresolved_contradiction": bool(getattr(critique, "has_unresolved_contradiction", False)),
        }

    @staticmethod
    def _decision_to_dict(decision: FinalAnswerDecision | None) -> dict[str, Any]:
        if decision is None:
            return {}
        return {
            "answer": str(getattr(decision, "answer", "")),
            "confidence": float(getattr(decision, "confidence", 0.0) or 0.0),
            "abstain": bool(getattr(decision, "abstain", False)),
            "reason": str(getattr(decision, "reason", "")),
        }

    def _fallback_report(
        self,
        question: str,
        state: ResearchState,
        stage: str,
        exc: Exception | None = None,
    ) -> str:
        evidence_preview = "\n".join(
            [
                f"- [{e.evidence_type}] {e.claim}: {e.excerpt[:220]}"
                for e in state.evidence[:6]
            ]
        ).strip()
        error_line = f"{type(exc).__name__}: {exc}" if exc is not None else stage
        return (
            "[Decision Status] fallback\n"
            "[Final Confidence] 0.0000\n"
            f"[Decision Reason] workflow_stage_failed:{stage}\n\n"
            f"Question: {question}\n"
            f"Workflow fallback at stage: {stage}\n"
            f"Error: {error_line}\n\n"
            f"Available evidence:\n{evidence_preview or 'N/A'}"
        )

    def _direct_inference_result(
        self,
        question: str,
        image_ref: str | None,
        route_mode: str,
    ) -> WorkflowResult:
        state = ResearchState(question=question)
        route_trace = {
            "requested_route": route_mode,
            "effective_route": "direct_inference",
            "short_circuit": True,
            "used_tools": False,
            "has_image": bool(str(image_ref or "").strip()),
        }
        try:
            prompt = (
                "Answer the user question directly and concisely. "
                "If the provided information is insufficient, say so instead of inventing details."
            )
            answer = self.synthesizer.llm.chat(
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": question},
                ],
                temperature=0.1,
                route="direct_inference",
            )
            decision = FinalAnswerDecision(
                answer=str(answer or "").strip(),
                confidence=0.35,
                abstain=False,
                reason="direct_inference_short_circuit",
            )
            report = (
                "[Decision Status] finalized\n"
                "[Final Confidence] 0.3500\n"
                "[Decision Reason] direct_inference_short_circuit\n\n"
                + decision.answer
            )
            return WorkflowResult(
                report=report,
                state=state,
                decision=decision,
                route_trace=route_trace,
                decision_status="finalized",
                final_confidence=0.35,
            )
        except Exception as exc:
            decision = FinalAnswerDecision(
                answer="",
                confidence=0.0,
                abstain=True,
                reason="direct_inference_exception",
            )
            return WorkflowResult(
                report=self._fallback_report(question, state, "direct_inference", exc),
                state=state,
                decision=decision,
                route_trace=route_trace,
                error_trace=[
                    {
                        "stage": "direct_inference",
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                ],
                decision_status="fallback",
                final_confidence=0.0,
            )

    def _structured_visual_summary(self, structured: dict) -> str:
        parts: list[str] = []

        visible_facts = structured.get("visible_facts", {})
        if isinstance(visible_facts, dict) and visible_facts:
            fact_lines = [f"{k}: {v}" for k, v in visible_facts.items()]
            parts.append("可见事实:\n" + "\n".join(fact_lines[:12]))

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
                parts.append("OCR事实:\n" + "\n".join(ocr_lines))

        ambiguities = structured.get("ambiguities", [])
        if isinstance(ambiguities, list) and ambiguities:
            amb = [str(x).strip() for x in ambiguities[:8] if str(x).strip()]
            if amb:
                parts.append("歧义:\n" + "\n".join(amb))

        hypotheses = structured.get("hypotheses", [])
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

    def _structured_visual_to_evidence(self, structured: dict, source: str = "local_image") -> list[Evidence]:
        out: list[Evidence] = []

        visible_facts = structured.get("visible_facts", {})
        if isinstance(visible_facts, dict) and visible_facts:
            fact_excerpt = "\n".join([f"{k}: {v}" for k, v in visible_facts.items()][:12])
            out.append(
                Evidence(
                    source=source,
                    claim="图像可见事实",
                    excerpt=fact_excerpt,
                    confidence=self._aggregate_visible_fact_confidence(visible_facts),
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
                conf = item.get("confidence", "")
                if text:
                    ocr_lines.append(f"[{region}] {text} (confidence={conf})")
            if ocr_lines:
                out.append(
                    Evidence(
                        source=f"{source} # ocr",
                        claim="图像 OCR 事实",
                        excerpt="\n".join(ocr_lines),
                        confidence=self._aggregate_ocr_confidence(ocr_facts),
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
                try:
                    conf = float(item.get("confidence", 0.55))
                except Exception:
                    conf = 0.55

                merged = f"{label} {reason}".strip()
                if not merged:
                    continue

                low = merged.lower()
                is_contradiction = any(
                    m in low
                    for m in [
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
                )

                out.append(
                    Evidence(
                        source=f"{source} # hypothesis",
                        claim=label or f"图像候选解释 {idx}",
                        excerpt=reason or merged[:500],
                        confidence=max(0.35, min(conf, 0.99)),
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
                        confidence=0.62,
                        evidence_type="ambiguity",
                        premise_dependencies=[],
                        ambiguity=0.95,
                        exclusivity=0.0,
                    )
                )

        return out

    def _seed_image_context(self, state: ResearchState, image_ref: str, question: str) -> str:
        """
        对输入图像做一次结构化分析，把结果注入 state，并返回扩展后的 planning question。
        """
        try:
            obs = self.researcher.tool_router.run(
                tool_name="analyze_image",
                args={
                    "image_ref": image_ref,
                    "question": question,
                    "fallback_text": "",
                },
            )
            state.tool_logs.append(obs)

            structured = self._safe_json_loads(obs.output_data)
            if structured:
                state.evidence.extend(self._structured_visual_to_evidence(structured, source="seed_local_image"))
                summary = self._structured_visual_summary(structured)
                self._compress_state_memory(state)

                if summary.strip():
                    return (
                        question
                        + "\n\n图像分析摘要:\n"
                        + summary
                        + "\n\n注意：后续应优先围绕这些可见事实、OCR 事实和歧义进行验证，"
                          "不要把候选解释直接当成已证实结论。"
                    )
        except Exception as exc:
            self._append_tool_error(
                state=state,
                tool_name="analyze_image",
                input_data={
                    "image_ref": image_ref,
                    "question": question,
                    "fallback_text": "",
                },
                exc=exc,
            )

        return question
    
    
    @staticmethod
    def _infer_reinspect_region_hints(question: str) -> list[str]:
        q = str(question or "").lower()
        hints: list[str] = []

        def _add(x: str) -> None:
            if x and x not in hints:
                hints.append(x)

        if any(k in q for k in ["text", "ocr", "label", "caption", "sign", "screen", "equation", "formula", "文字", "标签", "公式"]):
            _add("text")
        if any(k in q for k in ["table", "表格"]):
            _add("table")
        if any(k in q for k in ["chart", "graph", "plot", "axis", "legend", "bar", "line", "pie", "图表", "坐标", "曲线", "柱状图"]):
            _add("chart")
            _add("axis")
            _add("legend")
        if any(k in q for k in ["diagram", "geometry", "algebra", "triangle", "angle", "coordinate", "pair", "distance", "position", "direction", "几何", "代数", "位置", "方向", "距离"]):
            _add("diagram")
            _add("object_region")

        if not hints:
            _add("object_region")
            _add("text")

        return hints[:3]

    def _reinspect_image(self, state: ResearchState, image_ref: str, question: str) -> None:
        """
        当 critic 指出 must_reinspect_image=True 时，做一次题目驱动的局部重检。
        """
        region_hints = self._infer_reinspect_region_hints(question)

        for hint in region_hints:
            try:
                obs = self.researcher.tool_router.run(
                    tool_name="analyze_image_region",
                    args={
                        "image_ref": image_ref,
                        "question": question,
                        "region_hint": hint,
                        "fallback_text": "",
                    },
                )
                state.tool_logs.append(obs)
                structured = self._safe_json_loads(obs.output_data)
                if structured:
                    state.evidence.extend(
                        self._structured_visual_to_evidence(structured, source=f"reinspect:{hint}")
                    )
            except Exception as exc:
                self._append_tool_error(
                    state=state,
                    tool_name="analyze_image_region",
                    input_data={
                        "image_ref": image_ref,
                        "question": question,
                        "region_hint": hint,
                        "fallback_text": "",
                    },
                    exc=exc,
                )
                continue

            if hint in {"text", "table", "chart", "axis", "legend", "equation", "clock", "sign", "screen"}:
                try:
                    ocr_obs = self.researcher.tool_router.run(
                        tool_name="ocr_image_region",
                        args={
                            "image_ref": image_ref,
                            "region_hint": hint,
                            "fallback_text": "",
                        },
                    )
                    state.tool_logs.append(ocr_obs)
                    parsed = self._safe_json_loads(ocr_obs.output_data)
                    ocr_lines = parsed.get("ocr_lines", []) if isinstance(parsed, dict) else []
                    if isinstance(ocr_lines, list) and ocr_lines:
                        lines: list[str] = []
                        for item in ocr_lines[:10]:
                            if not isinstance(item, dict):
                                continue
                            text = str(item.get("text", "")).strip()
                            region = str(item.get("region", "unknown")).strip()
                            conf = item.get("confidence", "")
                            if text:
                                lines.append(f"[{region}] {text} (confidence={conf})")
                        if lines:
                            state.evidence.append(
                                Evidence(
                                    source=f"reinspect:{hint} # ocr",
                                    claim="局部OCR事实",
                                    excerpt="\n".join(lines),
                                    confidence=self._aggregate_ocr_confidence(ocr_lines),
                                    evidence_type="ocr",
                                    premise_dependencies=[],
                                    ambiguity=0.06,
                                    exclusivity=0.42,
                                )
                            )
                except Exception as exc:
                    self._append_tool_error(
                        state=state,
                        tool_name="ocr_image_region",
                        input_data={
                            "image_ref": image_ref,
                            "region_hint": hint,
                            "fallback_text": "",
                        },
                        exc=exc,
                    )

        self._compress_state_memory(state)

    def run_with_trace(
        self,
        question: str,
        image_ref: str | None = None,
        route_mode: str = "auto",
        seed_docs: list[dict[str, str]] | None = None,
    ) -> WorkflowResult:
        state = ResearchState(question=question)

        # 注册当前图像引用
        if image_ref:
            try:
                self.researcher.tool_router.set_current_image_ref(image_ref)
            except Exception:
                pass

        # seed 图像上下文，扩展 planning question
        planning_question = question
        if image_ref:
            planning_question = self._seed_image_context(
                state=state,
                image_ref=image_ref,
                question=question,
            )

        # planner -> postprocess_subquestions
        try:
            planned_sub_questions = self.planner.plan(planning_question)
        except Exception:
            planned_sub_questions = [SubQuestion(question=planning_question, intent="evidence", priority=1)]

        try:
            state.sub_questions = self.researcher.postprocess_subquestions(
                main_question=planning_question,
                sub_questions=planned_sub_questions,
                seed_logs=state.tool_logs,
            )
            if not state.sub_questions:
                state.sub_questions = [SubQuestion(question=planning_question, intent="evidence", priority=1)]
        except Exception:
            state.sub_questions = planned_sub_questions or [
                SubQuestion(question=planning_question, intent="evidence", priority=1)
            ]

        effective_route = (
            self._resolve_auto_route(
                question=question,
                image_ref=image_ref,
                seed_docs=seed_docs,
            )
            if str(route_mode or "").strip().lower() == "auto"
            else str(route_mode or "").strip().lower()
        ) or "direct_inference"

        route_trace = {
            "requested_route": str(route_mode or "").strip().lower() or "auto",
            "effective_route": effective_route,
            "has_image": bool(str(image_ref or "").strip()),
            "seed_document_count": len(seed_docs or []),
            "short_circuit": False,
        }
        critique_trace: list[dict[str, Any]] = []
        error_trace: list[dict[str, Any]] = []

        if effective_route == "direct_inference":
            return self._direct_inference_result(
                question=question,
                image_ref=image_ref,
                route_mode=route_mode,
            )

        last_critique = None
        fixed_seed_docs = list(seed_docs or [])

        # investigate -> critique -> follow-up / reinspect
        for _round_idx in range(max(1, int(getattr(settings, "max_rounds", 3) or 3))):
            current_questions = state.sub_questions[:]
            state.sub_questions = []

            if not current_questions:
                current_questions = [SubQuestion(question=planning_question, intent="fallback", priority=1)]

            for sq in current_questions:
                current_sq = sq
                used_qs = [str(sq.question).strip()]
                max_rewrites = max(
                    1,
                    int(getattr(settings, "max_subquestion_rewrites", 3) or 3),
                )

                accepted = False

                for _rewrite_idx in range(max_rewrites):
                    rolling_seed_docs = self._merge_seed_docs(
                        base_docs=fixed_seed_docs,
                        tool_logs=state.tool_logs,
                    )

                    ev, logs = self.researcher.investigate(
                        main_question=planning_question,
                        sub_question=current_sq,
                        seed_docs=rolling_seed_docs,
                        route_mode=effective_route,
                    )

                    state.evidence.extend(ev)
                    state.tool_logs.extend(logs)
                    self._compress_state_memory(state)

                    grounding_accept = True
                    grounding_reason = ""

                    for log in reversed(logs):
                        if log.tool_name == "subquestion_grounding_check":
                            parsed = self._safe_json_loads(log.output_data)
                            grounding_accept = bool(parsed.get("accept", True))
                            grounding_reason = str(parsed.get("reason", "")).strip()
                            break

                    if grounding_accept:
                        accepted = True
                        break

                    try:
                        rewritten = self.researcher._rewrite_subquestion_from_visual_feedback(
                            main_question=planning_question,
                            bad_sub_question=current_sq,
                            logs=state.tool_logs,
                            failure_reason=grounding_reason or "not_grounded_to_image",
                            used_questions=used_qs,
                        )
                    except Exception:
                        rewritten = current_sq

                    rewritten_q = str(getattr(rewritten, "question", "") or "").strip()
                    if not rewritten_q or rewritten_q in used_qs:
                        break

                    current_sq = rewritten
                    used_qs.append(rewritten_q)

                _ = accepted

            try:
                critique = self.critic.critique(question=question, evidence=state.evidence)
            except Exception as exc:
                error_trace.append(
                    {
                        "stage": "critic.critique",
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
                report = self._fallback_report(question, state, "critic.critique", exc)
                decision = FinalAnswerDecision(
                    answer="",
                    confidence=0.0,
                    abstain=True,
                    reason="critic_exception",
                )
                return WorkflowResult(
                    report=report,
                    state=state,
                    decision=decision,
                    critique_trace=critique_trace,
                    route_trace=route_trace,
                    error_trace=error_trace,
                    decision_status="fallback",
                    final_confidence=0.0,
                )
            last_critique = critique
            critique_trace.append({"stage": "round_critique", **self._critique_to_dict(critique)})

            if critique.sufficient:
                break

            if critique.must_reinspect_image and image_ref:
                self._reinspect_image(
                    state=state,
                    image_ref=image_ref,
                    question=question,
                )
                try:
                    critique = self.critic.critique(question=question, evidence=state.evidence)
                except Exception as exc:
                    error_trace.append(
                        {
                            "stage": "critic.critique_after_reinspect",
                            "type": type(exc).__name__,
                            "message": str(exc),
                        }
                    )
                    report = self._fallback_report(question, state, "critic.critique_after_reinspect", exc)
                    decision = FinalAnswerDecision(
                        answer="",
                        confidence=0.0,
                        abstain=True,
                        reason="critic_exception_after_reinspect",
                    )
                    return WorkflowResult(
                        report=report,
                        state=state,
                        decision=decision,
                        critique_trace=critique_trace,
                        route_trace=route_trace,
                        error_trace=error_trace,
                        decision_status="fallback",
                        final_confidence=0.0,
                    )
                last_critique = critique
                critique_trace.append({"stage": "reinspect_critique", **self._critique_to_dict(critique)})
                if critique.sufficient:
                    break

            if critique.must_abort_finalization:
                break

            if critique.follow_up_questions:
                state.sub_questions = [
                    SubQuestion(question=str(q).strip(), intent="gap_fill", priority=1)
                    for q in critique.follow_up_questions
                    if str(q).strip()
                ]
            else:
                state.sub_questions = [SubQuestion(question=question, intent="fallback", priority=1)]

        # verify -> final critique -> synthesize / decide
        try:
            state.evidence = self.verifier.verify(question=question, evidence=state.evidence)
        except Exception as exc:
            error_trace.append(
                {
                    "stage": "verifier.verify",
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            report = self._fallback_report(question, state, "verifier.verify", exc)
            decision = FinalAnswerDecision(
                answer="",
                confidence=0.0,
                abstain=True,
                reason="verifier_exception",
            )
            return WorkflowResult(
                report=report,
                state=state,
                decision=decision,
                critique_trace=critique_trace,
                route_trace=route_trace,
                error_trace=error_trace,
                decision_status="fallback",
                final_confidence=0.0,
            )

        try:
            final_critique = self.critic.critique(question=question, evidence=state.evidence)
            critique_trace.append({"stage": "final_critique", **self._critique_to_dict(final_critique)})
        except Exception as exc:
            error_trace.append(
                {
                    "stage": "critic.final_critique",
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            report = self._fallback_report(question, state, "critic.final_critique", exc)
            decision = FinalAnswerDecision(
                answer="",
                confidence=0.0,
                abstain=True,
                reason="final_critic_exception",
            )
            return WorkflowResult(
                report=report,
                state=state,
                decision=decision,
                critique_trace=critique_trace,
                route_trace=route_trace,
                error_trace=error_trace,
                decision_status="fallback",
                final_confidence=0.0,
            )

        try:
            report = self.synthesizer.synthesize(state)
        except Exception as exc:
            error_trace.append(
                {
                    "stage": "synthesizer.synthesize",
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            report = self._fallback_report(question, state, "synthesizer.synthesize", exc)

        try:
            decision = self.synthesizer.decide(state)
        except Exception as exc:
            error_trace.append(
                {
                    "stage": "synthesizer.decide",
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            decision = FinalAnswerDecision(
                answer="",
                confidence=float(getattr(final_critique, "final_confidence", 0.0) or 0.0),
                abstain=bool(getattr(final_critique, "must_abort_finalization", False)),
                reason="synthesizer_decide_exception",
            )

        if final_critique.must_abort_finalization and not final_critique.sufficient:
            prefix = (
                "[Decision Status] abstained_due_to_insufficient_evidence\n"
                f"[Dominant Failure Mode] {final_critique.dominant_failure_mode or 'insufficient_evidence'}\n"
                f"[Missing Dimensions] {', '.join(final_critique.missing_dimensions) if final_critique.missing_dimensions else 'N/A'}\n"
                f"[Final Confidence] {float(getattr(final_critique, 'final_confidence', 0.0)):.4f}\n\n"
            )
            final_report = prefix + report
            return WorkflowResult(
                report=final_report,
                state=state,
                decision=decision,
                final_critique=final_critique,
                critique_trace=critique_trace,
                route_trace=route_trace,
                error_trace=error_trace,
                decision_status="abstained_due_to_insufficient_evidence",
                final_confidence=float(getattr(final_critique, "final_confidence", 0.0) or 0.0),
            )

        status = "abstained" if decision.abstain else "finalized"
        reason_line = f"[Decision Reason] {decision.reason}\n\n" if decision.reason else "\n"

        final_report = (
            f"[Decision Status] {status}\n"
            f"[Final Confidence] {float(decision.confidence):.4f}\n"
            + reason_line
            + report
        )
        return WorkflowResult(
            report=final_report,
            state=state,
            decision=decision,
            final_critique=final_critique,
            critique_trace=critique_trace,
            route_trace=route_trace,
            error_trace=error_trace,
            decision_status=status,
            final_confidence=float(decision.confidence),
        )

    def run(
        self,
        question: str,
        image_ref: str | None = None,
        route_mode: str = "auto",
        seed_docs: list[dict[str, str]] | None = None,
    ) -> str:
        return self.run_with_trace(
            question=question,
            image_ref=image_ref,
            route_mode=route_mode,
            seed_docs=seed_docs,
        ).report
