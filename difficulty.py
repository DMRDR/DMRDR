from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any

from lexical_graph import LexicalGraphRetriever, build_graph_from_corpus


@dataclass
class DifficultyBreakdown:
    overall: float
    linguistic_complexity: float
    reasoning_complexity: float
    multimodal_complexity: float
    knowledge_intensity: float
    choice_ambiguity: float
    retrieval_hardness: float
    deepresearch_coverage: float


def _qnorm_100(v: float) -> float:
    return _clip(v) * 100.0


def _norm_linear(v: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return _clip((v - lo) / (hi - lo))


def _norm_q(v: float, q: float = 0.7) -> float:
    return _clip(_clip(v) ** max(q, 1e-6))


def _task_family(sample: dict[str, Any]) -> str:
    dataset_type = str(sample.get("dataset_type", "generic")).lower()
    if dataset_type in {"generic", "aokvqa", "a_okvqa", "okvqa", "fvqa", "krvqa", "cmmqa"}:
        return "knowledge_enhanced_vqa"
    return "ordinary_multimodal_vqa"


def _clip(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _is_missing(v: Any) -> bool:
    if v is None:
        return True
    try:
        # handles numpy/pandas NaN scalars safely
        return bool(v != v)
    except Exception:
        return False


def _first_non_missing(*vals: Any, default: Any = None) -> Any:
    for v in vals:
        if not _is_missing(v):
            return v
    return default


def _to_option_list(v: Any) -> list[Any]:
    if _is_missing(v):
        return []
    if hasattr(v, "tolist"):
        try:
            v = v.tolist()
        except Exception:
            pass
    if isinstance(v, (list, tuple)):
        return list(v)
    return []


def _reasoning_chain_paths(sample: dict[str, Any]) -> list[list[Any]]:
    rc = sample.get("reasoning_chains")
    if isinstance(rc, list):
        out: list[list[Any]] = []
        for item in rc:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                out.append(list(item))
        if out:
            return out
    docs = sample.get("documents", [])
    if isinstance(docs, list):
        for d in docs:
            if isinstance(d, dict) and str(d.get("source", "")) == "reasoning_chains":
                txt = str(d.get("text", ""))
                lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
                out = []
                for ln in lines:
                    if "--" in ln and "-->" in ln:
                        out.append([ln])
                if out:
                    return out
    return []


def _operator_profile(question: str, docs: Any) -> dict[str, float]:
    q = question.lower()
    ops = {
        "relate": sum(1 for k in ["related", "relation", "belong", "connected", "where", "which"] if k in q),
        "filter": sum(1 for k in ["except", "not", "without", "only", "at least", "at most"] if k in q),
        "andor": sum(1 for k in [" and ", " or ", "both", "either"] if k in q),
        "compare": sum(1 for k in ["more", "less", "than", "compare", "difference", "farthest", "nearest"] if k in q),
        "count": sum(1 for k in ["how many", "count", "number of"] if k in q),
    }
    if isinstance(docs, list):
        text = " ".join(str(d.get("text", "")).lower() for d in docs if isinstance(d, dict))
        if " if " in text or " then " in text:
            ops["andor"] += 1
    return {k: float(v) for k, v in ops.items()}


def _h_score(sample: dict[str, Any], reasoning_base: float) -> float:
    chains = _reasoning_chain_paths(sample)
    if chains:
        d_min = float(max(min(len(chains), 3), 1))
        return _norm_linear(d_min, 1.0, 3.0)

    dist_proxy = _to_float(_first_non_missing(sample.get("reasoning_hops"), sample.get("cot_steps"), sample.get("steps"), default=1), 1.0)
    docs = sample.get("documents", [])
    ls = 1.0
    if isinstance(docs, list):
        ls = max((str(d.get("text", "")).count("\n") + 1 for d in docs if isinstance(d, dict)), default=1)
    h_raw = 0.6 * dist_proxy + 0.4 * ls
    return _clip(0.6 * _norm_linear(h_raw, 1.0, 5.0) + 0.4 * reasoning_base)


def _d_score(sample: dict[str, Any], ambiguity: float, retrieval_hardness: float, multimodal: float) -> float:
    chains = _reasoning_chain_paths(sample)
    if chains:
        branch = {}
        for c in chains:
            if len(c) >= 3:
                h = str(c[0]).strip().lower()
                t = str(c[2]).strip().lower()
                branch[h] = branch.get(h, 0) + 1
                branch[t] = branch.get(t, 0) + 1
        if branch:
            deg_avg = sum(max(v - 1, 0) for v in branch.values()) / max(len(branch), 1)
            return _norm_q(_norm_linear(deg_avg, 0.0, 4.0), q=0.8)

    options = _to_option_list(_first_non_missing(sample.get("choices"), sample.get("options"), default=[]))
    text = str(sample.get("question", "")).lower()
    cand_proxy = len(options) + sum(1 for k in ["who", "which", "what", "where"] if k in text)
    d_text = _norm_linear(cand_proxy - 1.0, 0.0, 10.0)
    d_vis = _clip(multimodal)
    return _clip(0.45 * d_text + 0.35 * ambiguity + 0.20 * max(retrieval_hardness, d_vis))


def _o_score(sample: dict[str, Any], linguistic: float) -> float:
    ops = _operator_profile(str(sample.get("question", "")), sample.get("documents", []))
    weights = {"relate": 0.20, "filter": 0.20, "andor": 0.15, "compare": 0.25, "count": 0.20}
    raw = sum(weights[k] * ops.get(k, 0.0) for k in weights)
    op_complex = _norm_q(_norm_linear(raw, 0.0, 4.0), q=0.75)
    return _clip(0.65 * op_complex + 0.35 * linguistic)


def _a_score(multimodal: float, coverage: float, knowledge: float) -> float:
    a_vis = _clip(multimodal)
    a_cov = _clip(1.0 - coverage)
    a_kg_dep = _clip(knowledge)
    a_raw = 0.45 * a_vis + 0.30 * a_cov + 0.25 * a_kg_dep
    return _norm_q(a_raw, q=0.85)


def _combine_hdoa(sample: dict[str, Any], h: float, d: float, o: float, a: float) -> float:
    fam = _task_family(sample)
    if fam == "knowledge_enhanced_vqa":
        w_h, w_d, w_o, w_a = 0.20, 0.25, 0.10, 0.35
    else:
        w_h, w_d, w_o, w_a = 0.30, 0.25, 0.15, 0.30
    return _clip(w_h * h + w_d * d + w_o * o + w_a * a)


def _has_image(sample: dict[str, Any]) -> tuple[bool, int]:
    image_url = sample.get("image_url")
    images = sample.get("images")
    has_image = isinstance(image_url, str) and bool(image_url.strip())
    image_count = 0
    if isinstance(images, list):
        image_count = len(images)
        has_image = has_image or image_count > 0
    elif hasattr(images, "tolist"):
        try:
            img_list = images.tolist()
            if isinstance(img_list, list):
                image_count = len(img_list)
                has_image = has_image or image_count > 0
        except Exception:
            pass
    if isinstance(sample.get("image"), str) and sample.get("image", "").strip():
        has_image = True
        image_count = max(image_count, 1)
    if isinstance(sample.get("image_path"), str) and sample.get("image_path", "").strip():
        has_image = True
        image_count = max(image_count, 1)
    if isinstance(sample.get("img_path"), str) and sample.get("img_path", "").strip():
        has_image = True
        image_count = max(image_count, 1)
    if isinstance(sample.get("__image_bytes"), (bytes, bytearray)):
        has_image = True
        image_count = max(image_count, 1)
    return has_image, image_count


def _base_components(sample: dict[str, Any]) -> tuple[float, float, float, float, float, float, float]:
    question = str(sample.get("question", ""))
    q_len = len(question)
    linguistic = _clip((q_len - 20) / 120)

    reasoning_hints = _first_non_missing(
        sample.get("reasoning_hops"), sample.get("cot_steps"), sample.get("steps"), default=1
    )
    try:
        hops = float(reasoning_hints)
    except Exception:
        hops = 1.0
    reasoning = _clip((hops - 1) / 5)

    has_image, image_count = _has_image(sample)
    table_flag = 1.0 if sample.get("has_table") else 0.0
    chart_flag = 1.0 if sample.get("has_chart") else 0.0
    multimodal = _clip(0.35 * (1 if has_image else 0) + 0.2 * min(image_count, 3) / 3 + 0.25 * table_flag + 0.2 * chart_flag)

    docs = sample.get("documents", [])
    doc_count = len(docs) if isinstance(docs, list) else 0
    domain = str(sample.get("domain", "")).lower()
    domain_bonus = 0.15 if domain in {"science", "medicine", "law", "finance", "engineering"} else 0.0
    knowledge = _clip(min(doc_count, 8) / 8 + domain_bonus)

    options = _to_option_list(_first_non_missing(sample.get("choices"), sample.get("options"), default=[]))
    ambiguity = _clip((len(options) - 2) / 6) if options else 0.15

    retrieval_hardness, coverage = _estimate_retrieval_hardness_and_coverage(question=question, docs=docs)
    return linguistic, reasoning, multimodal, knowledge, ambiguity, retrieval_hardness, coverage


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _to_token_list(text: str) -> list[str]:
    return [t for t in text.strip().split() if t]


def _detect_m3cot_question_type(question: str) -> str:
    q = question.lower()
    if any(k in q for k in ["how many", "数量", "几", "count"]):
        return "count"
    if any(k in q for k in ["why", "原因", "because", "推断", "infer"]):
        return "reasoning"
    if any(k in q for k in ["what color", "颜色", "where", "位置", "which"]):
        return "attribute"
    return "generic"


def _m3cot_vision_score(sample: dict[str, Any]) -> float:
    obj_count = _to_float(_first_non_missing(sample.get("object_count"), sample.get("num_objects"), default=0), 0.0)
    occlusion = _to_float(sample.get("occlusion_ratio"), 0.0)
    small_ratio = _to_float(sample.get("small_object_ratio"), 0.0)
    ocr_chars = _to_float(_first_non_missing(sample.get("ocr_char_count"), sample.get("text_char_count"), default=0), 0.0)
    ocr_area_ratio = _to_float(sample.get("ocr_area_ratio"), 0.0)

    if obj_count <= 0:
        # fallback: infer rough object complexity from image+caption/doc presence
        has_image, _ = _has_image(sample)
        obj_count = 4.0 if has_image else 0.0
    if ocr_chars <= 0:
        docs = sample.get("documents", [])
        if isinstance(docs, list):
            joined = " ".join(str(d.get("text", "")) for d in docs if isinstance(d, dict))
            ocr_chars = min(len(joined), 120)

    s_obj = _clip(min(obj_count, 20.0) / 20.0) * 100.0
    s_occ = _clip(occlusion / 0.8) * 100.0
    s_small = _clip(small_ratio / 0.5) * 100.0
    s_text = _clip(min((ocr_area_ratio * 2.0) + (ocr_chars / 120.0), 1.0)) * 100.0
    return _clip(mean([s_obj, s_occ, s_small, s_text]) / 100.0) * 100.0


def _m3cot_question_score(sample: dict[str, Any]) -> float:
    question = str(sample.get("question", ""))
    tokens = _to_token_list(question)
    q_len = len(tokens)
    q_len_score = _clip(min(q_len, 30) / 30.0) * 100.0

    pronouns = sum(1 for t in tokens if t.lower() in {"it", "this", "that", "these", "those", "they"})
    pronoun_ratio = pronouns / max(q_len, 1)
    s_amb = _clip(pronoun_ratio / 0.15) * 100.0

    q_lower = question.lower()
    op_hits = sum(1 for k in ["compare", "difference", "both", "and", "or", "if", "then", "more", "less"] if k in q_lower)
    s_ops = _clip(op_hits / 4.0) * 100.0

    span_hits = sum(1 for k in ["text", "number", "word", "table", "chart", "color", "shape", "位置", "数字"] if k in q_lower)
    s_span = _clip(span_hits / 4.0) * 100.0
    return _clip(mean([q_len_score, s_amb, s_ops, s_span]) / 100.0) * 100.0


def _m3cot_reasoning_score(sample: dict[str, Any]) -> float:
    question = str(sample.get("question", ""))
    q_type = _detect_m3cot_question_type(question)
    occlusion = _clip(_to_float(sample.get("occlusion_ratio"), 0.1) / 0.8) * 100.0
    small_ratio = _clip(_to_float(sample.get("small_object_ratio"), 0.1) / 0.5) * 100.0
    cluster = _clip(_to_float(_first_non_missing(sample.get("cluster_count"), sample.get("object_cluster_count"), default=1), 1.0) / 10.0) * 100.0

    if q_type == "count":
        return _clip((0.4 * occlusion + 0.3 * small_ratio + 0.3 * cluster) / 100.0) * 100.0
    if q_type == "reasoning":
        rationale = str(sample.get("rationale", ""))
        steps = max(rationale.count("\n") + 1, 1) if rationale.strip() else 1
        step_score = _clip((steps - 1) / 8.0) * 100.0
        return _clip((0.5 * step_score + 0.3 * occlusion + 0.2 * cluster) / 100.0) * 100.0
    return _clip((0.4 * cluster + 0.3 * occlusion + 0.3 * small_ratio) / 100.0) * 100.0


def _m3cot_e_model_score(sample: dict[str, Any]) -> float:
    # expected optional field from external baseline ensemble
    arr = sample.get("model_errors")
    if isinstance(arr, list) and arr:
        vals = [_to_float(x, 0.0) for x in arr]
        avg = sum(vals) / max(len(vals), 1)
        # 1 means error, 0 means correct
        return _clip(avg) * 100.0

    baseline_correct = sample.get("baseline_correct")
    if baseline_correct is not None:
        try:
            return (0.0 if bool(baseline_correct) else 100.0)
        except Exception:
            pass

    # fallback proxy by choice ambiguity + retrieval hardness
    options = _to_option_list(_first_non_missing(sample.get("choices"), sample.get("options"), default=[]))
    opt_factor = _clip((len(options) - 2) / 8.0) * 100.0 if options else 30.0
    docs = sample.get("documents", [])
    hardness, _ = _estimate_retrieval_hardness_and_coverage(str(sample.get("question", "")), docs)
    return _clip((0.6 * opt_factor + 0.4 * hardness * 100.0) / 100.0) * 100.0


def _aok_visual_complexity(sample: dict[str, Any]) -> float:
    obj = _to_float(_first_non_missing(sample.get("object_count"), sample.get("num_objects"), sample.get("obj_term"), default=0.0), 0.0)
    texture = _to_float(_first_non_missing(sample.get("texture_contrast"), sample.get("edge_density"), default=0.0), 0.0)
    text_area = _to_float(_first_non_missing(sample.get("ocr_area_ratio"), sample.get("text_density"), default=0.0), 0.0)
    has_image, img_cnt = _has_image(sample)

    if obj <= 0 and has_image:
        obj = 5.0
    obj_n = _clip(obj / 20.0) if obj > 1.0 else _clip(obj)
    texture_n = _clip(texture / 0.8) if texture > 1.0 else _clip(texture)
    text_n = _clip(text_area / 0.8) if text_area > 1.0 else _clip(text_area)
    img_n = _clip((img_cnt - 1) / 3.0)
    return _clip(0.45 * obj_n + 0.25 * texture_n + 0.20 * text_n + 0.10 * img_n)


def _aok_knowledge_demand(sample: dict[str, Any]) -> float:
    question = str(sample.get("question", ""))
    answer = str(sample.get("answer", ""))
    text = f"{question} {answer}".lower()
    tokens = _to_token_list(text)
    token_count = len(tokens)

    technical_hits = sum(1 for k in ["biology", "physics", "chemistry", "law", "finance", "medical", "历史", "地理", "数学"] if k in text)
    long_token_hits = sum(1 for t in tokens if len(t) >= 9)
    docs = sample.get("documents", [])
    doc_factor = _clip((len(docs) if isinstance(docs, list) else 0) / 8.0)
    rarity_proxy = _clip((technical_hits + long_token_hits / max(token_count, 1) * 4.0) / 6.0)
    return _clip(0.55 * rarity_proxy + 0.45 * doc_factor)


def _aok_reasoning_complexity(sample: dict[str, Any]) -> float:
    q = str(sample.get("question", "")).lower()
    base = _clip(_to_float(_first_non_missing(sample.get("reasoning_hops"), sample.get("cot_steps"), sample.get("steps"), default=1.0), 1.0) / 6.0)
    op_hits = sum(1 for k in ["why", "because", "compare", "difference", "both", "except", "if", "then", "most", "least", "推断", "比较"] if k in q)
    op_factor = _clip(op_hits / 6.0)
    options = _to_option_list(_first_non_missing(sample.get("choices"), sample.get("options"), default=[]))
    option_factor = _clip((len(options) - 2) / 8.0) if options else 0.2
    return _clip(0.45 * base + 0.35 * op_factor + 0.20 * option_factor)


def _aok_emodel_difficulty(sample: dict[str, Any]) -> float:
    preds = sample.get("model_predictions")
    gold = str(sample.get("answer", "")).strip().lower()
    if isinstance(preds, list) and preds:
        total = 0
        correct = 0
        for p in preds:
            s = str(p).strip().lower()
            if not s:
                continue
            total += 1
            if s == gold:
                correct += 1
        if total > 0:
            return _clip(1.0 - (correct / total))

    b_exact = sample.get("baseline_exact")
    if b_exact is not None:
        try:
            return 0.0 if bool(b_exact) else 1.0
        except Exception:
            pass

    # fallback with retrieval hardness proxy
    docs = sample.get("documents", [])
    hardness, _ = _estimate_retrieval_hardness_and_coverage(str(sample.get("question", "")), docs)
    return _clip(0.5 + 0.5 * hardness)


def _mmmu_type_prior(sample: dict[str, Any]) -> float:
    img_type = str(_first_non_missing(sample.get("img_type"), sample.get("image_type"), sample.get("category"), default="")).lower()
    if any(k in img_type for k in ["chart", "graph", "plot"]):
        return 0.8
    if any(k in img_type for k in ["table", "document", "text"]):
        return 0.75
    if any(k in img_type for k in ["diagram", "map", "flow"]):
        return 0.7
    if img_type:
        return 0.65
    return 0.6


def _mmmu_cvision(sample: dict[str, Any]) -> float:
    obj = _to_float(_first_non_missing(sample.get("obj_term"), sample.get("object_density"), sample.get("num_objects"), default=0.0), 0.0)
    small = _to_float(_first_non_missing(sample.get("small_term"), sample.get("small_object_ratio"), default=0.0), 0.0)
    occ = _to_float(_first_non_missing(sample.get("occ_term"), sample.get("occlusion_ratio"), default=0.0), 0.0)
    text_density = _to_float(sample.get("text_density"), 0.0)
    n_images = _to_float(_first_non_missing(sample.get("n_images"), sample.get("image_count"), default=1.0), 1.0)

    # normalize signals to 0..1
    obj_n = _clip(obj / 20.0) if obj > 1.0 else _clip(obj)
    small_n = _clip(small / 0.5) if small > 1.0 else _clip(small)
    occ_n = _clip(occ / 0.8) if occ > 1.0 else _clip(occ)
    text_n = _clip(text_density / 8.0) if text_density > 1.0 else _clip(text_density)
    multi_n = _clip((n_images - 1.0) / 3.0)
    type_prior = _mmmu_type_prior(sample)

    return _clip(0.25 * obj_n + 0.15 * small_n + 0.10 * occ_n + 0.20 * text_n + 0.05 * multi_n + 0.25 * type_prior)


def _mmmu_cquestion(sample: dict[str, Any]) -> float:
    question = str(sample.get("question", ""))
    tokens = _to_token_list(question)
    q_len = len(tokens)
    len_tokens = _clip(q_len / 40.0)

    q_lower = question.lower()
    op_cnt = sum(1 for k in ["compare", "difference", "both", "and", "or", "if", "then", "more", "less", "choose"] if k in q_lower)
    op_term = _clip(op_cnt / 6.0)
    num_sym = _clip(sum(ch.isdigit() for ch in question) / max(len(question), 1) / 0.1)

    options = _to_option_list(_first_non_missing(sample.get("choices"), sample.get("options"), default=[]))
    option_count = len(options)
    option_term = _clip((option_count - 2) / 8.0) if option_count else 0.4

    if option_count >= 2:
        lengths = [len(str(o)) for o in options]
        len_gap = (max(lengths) - min(lengths)) if lengths else 0
    else:
        len_gap = 0
    len_gap_term = _clip(1.0 - (len_gap / max(sum(len(str(o)) for o in options), 1))) if options else 1.0

    # similarity fallback proxy (without external embedding model): overlap of option tokens
    if option_count >= 2:
        token_sets = [set(_to_token_list(str(o).lower())) for o in options]
        sims: list[float] = []
        for i in range(len(token_sets)):
            for j in range(i + 1, len(token_sets)):
                a, b = token_sets[i], token_sets[j]
                if not a and not b:
                    sims.append(1.0)
                else:
                    sims.append(len(a & b) / max(len(a | b), 1))
        sim_mean = sum(sims) / max(len(sims), 1)
        sim_max = max(sims) if sims else 0.0
    else:
        sim_mean = 0.0
        sim_max = 0.0

    return _clip(
        0.18 * len_tokens
        + 0.18 * op_term
        + 0.10 * num_sym
        + 0.18 * sim_mean
        + 0.18 * sim_max
        + 0.12 * option_term
        + 0.06 * len_gap_term
    )


def _mmmu_depth(sample: dict[str, Any]) -> float:
    question = str(sample.get("question", ""))
    q_lower = question.lower()
    lang_step = 1.0 if any(k in q_lower for k in ["first", "then", "finally", "步骤", "首先", "然后"]) else 0.0
    n_images = _to_float(_first_non_missing(sample.get("n_images"), sample.get("image_count"), default=1.0), 1.0)
    multi_img = 1.0 if n_images > 1 else 0.0

    img_type = str(_first_non_missing(sample.get("img_type"), sample.get("image_type"), sample.get("category"), default="")).lower()
    text_read = 1.0 if any(k in img_type for k in ["chart", "graph", "table", "document", "text"]) else 0.0
    text_density = _to_float(sample.get("text_density"), 0.0)
    if text_density > 2.0:
        text_read = max(text_read, 1.0)

    return _clip((lang_step + 0.6 * multi_img + 0.6 * text_read) / 4.0)


def _mmmu_creasoning(sample: dict[str, Any]) -> float:
    question = str(sample.get("question", "")).lower()
    if any(k in question for k in ["chart", "graph", "trend", "plot"]):
        return 0.70
    if any(k in question for k in ["table", "row", "column", "cell"]):
        return 0.66
    if any(k in question for k in ["infer", "why", "reason", "because", "推断", "原因"]):
        return 0.72
    if any(k in question for k in ["count", "how many", "数量", "几"]):
        return 0.62
    return 0.58


def _mmmu_emodel(sample: dict[str, Any]) -> float:
    # expected signals: prediction accuracy/correct flags; missing => hardest (1.0)
    acc = _first_non_missing(sample.get("model_accuracy"), sample.get("prediction_accuracy"), default=None)
    if acc is not None:
        return _clip(1.0 - _to_float(acc, 0.0))

    preds = sample.get("model_predictions")
    gold = str(sample.get("answer", "")).strip().lower()
    if isinstance(preds, list) and preds:
        correct = 0
        total = 0
        for p in preds:
            ps = str(p).strip().lower()
            if not ps:
                continue
            total += 1
            if ps == gold:
                correct += 1
        if total > 0:
            return _clip(1.0 - (correct / total))

    baseline_correct = sample.get("baseline_correct")
    if baseline_correct is not None:
        try:
            return 0.0 if bool(baseline_correct) else 1.0
        except Exception:
            pass
    return 1.0


def _mmmu_topic_calibration(sample: dict[str, Any]) -> float | None:
    raw = str(_first_non_missing(sample.get("topic_difficulty"), sample.get("difficulty_level"), default="")).strip().lower()
    if not raw:
        return None
    if raw in {"easy", "low"}:
        return 0.35
    if raw in {"medium", "med", "mid"}:
        return 0.65
    if raw in {"hard", "high"}:
        return 0.85
    return None


def _estimate_aokvqa_difficulty(sample: dict[str, Any]) -> DifficultyBreakdown:
    c_vision = _aok_visual_complexity(sample)
    c_knowledge = _aok_knowledge_demand(sample)
    c_reasoning = _aok_reasoning_complexity(sample)

    docs = sample.get("documents", [])
    retrieval_hardness, coverage = _estimate_retrieval_hardness_and_coverage(str(sample.get("question", "")), docs)
    options = _to_option_list(_first_non_missing(sample.get("choices"), sample.get("options"), default=[]))
    ambiguity = _clip((len(options) - 2) / 6) if options else 0.15

    H = _h_score(sample, reasoning_base=c_reasoning)
    D = _d_score(sample, ambiguity=ambiguity, retrieval_hardness=retrieval_hardness, multimodal=c_vision)
    O = _o_score(sample, linguistic=_clip(0.65 * c_knowledge + 0.35 * ambiguity))
    A = _a_score(multimodal=c_vision, coverage=coverage, knowledge=c_knowledge)
    overall = _combine_hdoa(sample, H, D, O, A)

    return DifficultyBreakdown(
        overall,
        linguistic_complexity=O,
        reasoning_complexity=H,
        multimodal_complexity=A,
        knowledge_intensity=c_knowledge,
        choice_ambiguity=ambiguity,
        retrieval_hardness=retrieval_hardness,
        deepresearch_coverage=coverage,
    )


def _estimate_m3cot_difficulty(sample: dict[str, Any]) -> DifficultyBreakdown:
    vision_score = _m3cot_vision_score(sample) / 100.0
    question_score = _m3cot_question_score(sample) / 100.0
    reasoning_score = _m3cot_reasoning_score(sample) / 100.0
    e_model_score = _m3cot_e_model_score(sample) / 100.0

    docs = sample.get("documents", [])
    retrieval_hardness, coverage = _estimate_retrieval_hardness_and_coverage(str(sample.get("question", "")), docs)
    options = _to_option_list(_first_non_missing(sample.get("choices"), sample.get("options"), default=[]))
    ambiguity = _clip((len(options) - 2) / 6) if options else 0.15

    H = _h_score(sample, reasoning_base=reasoning_score)
    D = _d_score(sample, ambiguity=ambiguity, retrieval_hardness=retrieval_hardness, multimodal=vision_score)
    O = _o_score(sample, linguistic=question_score)
    A = _a_score(multimodal=_clip(0.6 * vision_score + 0.4 * question_score), coverage=coverage, knowledge=e_model_score)
    overall = _combine_hdoa(sample, H, D, O, A)

    return DifficultyBreakdown(
        overall,
        linguistic_complexity=O,
        reasoning_complexity=H,
        multimodal_complexity=A,
        knowledge_intensity=e_model_score,
        choice_ambiguity=ambiguity,
        retrieval_hardness=retrieval_hardness,
        deepresearch_coverage=coverage,
    )


def _estimate_mmmu_difficulty(sample: dict[str, Any]) -> DifficultyBreakdown:
    cvision = _mmmu_cvision(sample)
    cquestion = _mmmu_cquestion(sample)
    depth = _mmmu_depth(sample)
    creasoning = _mmmu_creasoning(sample)
    emodel = _mmmu_emodel(sample)

    options = _to_option_list(_first_non_missing(sample.get("choices"), sample.get("options"), default=[]))
    ambiguity = _clip((len(options) - 2) / 6) if options else 0.15
    docs = sample.get("documents", [])
    retrieval_hardness, coverage = _estimate_retrieval_hardness_and_coverage(str(sample.get("question", "")), docs)

    H = _h_score(sample, reasoning_base=depth)
    D = _d_score(sample, ambiguity=ambiguity, retrieval_hardness=retrieval_hardness, multimodal=cvision)
    O = _o_score(sample, linguistic=cquestion)
    A = _a_score(multimodal=_clip(0.6 * cvision + 0.4 * cquestion), coverage=coverage, knowledge=emodel)
    base = _combine_hdoa(sample, H, D, O, A)
    calib = _mmmu_topic_calibration(sample)
    overall = _clip(0.8 * base + 0.2 * calib) if calib is not None else base

    return DifficultyBreakdown(
        overall=overall,
        linguistic_complexity=O,
        reasoning_complexity=H,
        multimodal_complexity=A,
        knowledge_intensity=emodel,
        choice_ambiguity=ambiguity,
        retrieval_hardness=retrieval_hardness,
        deepresearch_coverage=coverage,
    )


def estimate_question_difficulty(sample: dict[str, Any]) -> DifficultyBreakdown:
    """Dataset-aware difficulty estimator for AOKVQA/MMMU-Pro/M3CoT."""

    dataset_type = str(sample.get("dataset_type", "generic")).lower()
    if dataset_type in {"m3cot"}:
        return _estimate_m3cot_difficulty(sample)
    if dataset_type in {"mmmu_pro", "mmmu"}:
        return _estimate_mmmu_difficulty(sample)
    if dataset_type in {"generic", "aokvqa", "a_okvqa", "cmmqa", "scienceqa"}:
        return _estimate_aokvqa_difficulty(sample)
    return _estimate_aokvqa_difficulty(sample)


def _estimate_retrieval_hardness_and_coverage(question: str, docs: Any) -> tuple[float, float]:
    if not isinstance(docs, list):
        return 0.85, 0.0

    corpus: list[tuple[str, str]] = []
    for d in docs:
        if not isinstance(d, dict):
            continue
        source = str(d.get("source", "unknown"))
        text = str(d.get("text", "")).strip()
        if text:
            corpus.append((source, text))

    if not corpus:
        return 0.9, 0.0

    index = build_graph_from_corpus(corpus)
    hits = LexicalGraphRetriever(index).retrieve(question, top_k=5)
    if not hits:
        return 0.9, 0.0

    scores = [h.score for h in hits]
    avg_score = mean(scores)
    hardness = _clip(1.0 / (1.0 + avg_score))

    sources = {h.source for h in hits}
    total_sources = len({s for s, _ in corpus})
    coverage = _clip(len(sources) / max(total_sources, 1))
    return hardness, coverage


def difficulty_bucket(v: float, q1: float = 0.34, q2: float = 0.67) -> str:
    if v <= q1:
        return "easy"
    if v <= q2:
        return "medium"
    return "hard"
