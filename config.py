from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    """Runtime configuration for a DeepResearch-like Qwen3-VL framework."""

    model_name: str = field(default_factory=lambda: os.getenv("MODEL_NAME", "Qwen/Qwen3-VL-8B-Instruct"))
    api_base: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"))
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", "EMPTY"))

    api_timeout_s: float = field(default_factory=lambda: float(os.getenv("API_TIMEOUT_S", "60")))
    api_retries: int = field(default_factory=lambda: int(os.getenv("API_RETRIES", "3")))
    api_retry_backoff_s: float = field(default_factory=lambda: float(os.getenv("API_RETRY_BACKOFF_S", "2")))
    api_retry_max_wait_s: float = field(
        default_factory=lambda: float(os.getenv("API_RETRY_MAX_WAIT_S", "60"))
    )
    api_max_tokens: int = field(default_factory=lambda: int(os.getenv("API_MAX_TOKENS", "512")))

    max_rounds: int = field(default_factory=lambda: int(os.getenv("MAX_RESEARCH_ROUNDS", "3")))
    max_snippets_per_round: int = field(default_factory=lambda: int(os.getenv("MAX_SNIPPETS_PER_ROUND", "5")))
    max_actions_per_question: int = field(default_factory=lambda: int(os.getenv("MAX_ACTIONS_PER_QUESTION", "4")))
    max_evidence_items: int = field(default_factory=lambda: int(os.getenv("MAX_EVIDENCE_ITEMS", "12")))
    max_tool_logs: int = field(default_factory=lambda: int(os.getenv("MAX_TOOL_LOGS", "16")))

    search_api_url: str | None = field(default_factory=lambda: os.getenv("SEARCH_API_URL"))
    search_api_key: str | None = field(default_factory=lambda: os.getenv("SEARCH_API_KEY"))
    
    # ===== Embedding / Rerank =====

    enable_embedding_rerank: bool = field(
        default_factory=lambda: os.getenv("ENABLE_EMBEDDING_RERANK", "1").strip().lower() in {"1", "true", "yes", "on"}
    )
    embedding_api_base: str = field(default_factory=lambda: os.getenv("EMBEDDING_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
    embedding_api_key: str = field(default_factory=lambda: os.getenv("EMBEDDING_API_KEY", "sk-"))
    embedding_model_name: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-v4"))
    embedding_rerank_keep_k: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_RERANK_KEEP_K", "8"))
    )
    embedding_batch_size: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_BATCH_SIZE", "8"))
    )

    enable_qwen_chat_rerank: bool = field(
        default_factory=lambda: os.getenv("ENABLE_QWEN_CHAT_RERANK", "1").strip().lower() in {"1", "true", "yes", "on"}
    )
    rerank_api_base: str = field(default_factory=lambda: os.getenv("RERANK_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
    rerank_api_key: str = field(default_factory=lambda: os.getenv("RERANK_API_KEY", "sk-"))
    rerank_chat_model_name: str = field(default_factory=lambda: os.getenv("RERANK_CHAT_MODEL_NAME", "qwen3-rerank"))
    rerank_chat_top_n: int = field(
        default_factory=lambda: int(os.getenv("RERANK_CHAT_TOP_N", "4"))
    )
    
    # ===== CLIP Recall + BGE Rerank + MLLM =====
    enable_clip_rerank_mllm: bool = field(
        default_factory=lambda: os.getenv("ENABLE_CLIP_RERANK_MLLM", "0").strip().lower() in {"1", "true", "yes", "on"}
    )
    
    enable_clip_rerank_progress: bool = field(
        default_factory=lambda: os.getenv("ENABLE_CLIP_RERANK_PROGRESS", "1").strip().lower() in {"1", "true", "yes", "on"}
    )
    clip_rerank_progress_leave: bool = field(
        default_factory=lambda: os.getenv("CLIP_RERANK_PROGRESS_LEAVE", "0").strip().lower() in {"1", "true", "yes", "on"}
    )

    clip_model_name: str = field(
        default_factory=lambda: os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")
    )
    clip_device: str = field(
        default_factory=lambda: os.getenv("CLIP_DEVICE", "auto")
    )
    clip_local_files_only: bool = field(
        default_factory=lambda: os.getenv("CLIP_LOCAL_FILES_ONLY", "0").strip().lower() in {"1", "true", "yes", "on"}
    )
    clip_recall_top_k: int = field(
        default_factory=lambda: int(os.getenv("CLIP_RECALL_TOP_K", "10"))
    )
    clip_context_top_k: int = field(
        default_factory=lambda: int(os.getenv("CLIP_CONTEXT_TOP_K", "6"))
    )
    clip_batch_size: int = field(
        default_factory=lambda: int(os.getenv("CLIP_BATCH_SIZE", "8"))
    )
    clip_doc_max_chars: int = field(
        default_factory=lambda: int(os.getenv("CLIP_DOC_MAX_CHARS", "1200"))
    )

    enable_bge_reranker: bool = field(
        default_factory=lambda: os.getenv("ENABLE_BGE_RERANKER", "1").strip().lower() in {"1", "true", "yes", "on"}
    )
    bge_reranker_model_name: str = field(
        default_factory=lambda: os.getenv("BGE_RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
    )
    bge_reranker_device: str = field(
        default_factory=lambda: os.getenv("BGE_RERANKER_DEVICE", os.getenv("CLIP_DEVICE", "auto"))
    )
    bge_reranker_batch_size: int = field(
        default_factory=lambda: int(os.getenv("BGE_RERANKER_BATCH_SIZE", "8"))
    )
    
    # ===== Self-RAG + MLLM =====

    enable_self_rag_mllm: bool = field(
        default_factory=lambda: os.getenv("ENABLE_SELF_RAG_MLLM", "0").strip().lower() in {"1", "true", "yes", "on"}
    )

    # adaptive_retrieval / no_retrieval / always_retrieve
    self_rag_mllm_mode: str = field(
        default_factory=lambda: os.getenv("SELF_RAG_MLLM_MODE", "adaptive_retrieval").strip().lower()
    )

    self_rag_max_steps: int = field(
        default_factory=lambda: int(os.getenv("SELF_RAG_MAX_STEPS", "3"))
    )

    self_rag_retrieval_top_k: int = field(
        default_factory=lambda: int(os.getenv("SELF_RAG_RETRIEVAL_TOP_K", "5"))
    )

    self_rag_min_relevance: float = field(
        default_factory=lambda: float(os.getenv("SELF_RAG_MIN_RELEVANCE", "0.45"))
    )

    self_rag_support_threshold: float = field(
        default_factory=lambda: float(os.getenv("SELF_RAG_SUPPORT_THRESHOLD", "0.60"))
    )

    self_rag_context_max_chars: int = field(
        default_factory=lambda: int(os.getenv("SELF_RAG_CONTEXT_MAX_CHARS", "5000"))
    )

    self_rag_no_web_search: bool = field(
        default_factory=lambda: os.getenv("SELF_RAG_NO_WEB_SEARCH", "1").strip().lower() in {"1", "true", "yes", "on"}
    )
    
    # ===== mR2AG + MLLM =====

    enable_mr2ag_mllm: bool = field(
        default_factory=lambda: os.getenv("ENABLE_MR2AG_MLLM", "0").strip().lower() in {"1", "true", "yes", "on"}
    )

    # auto / no_retrieval / always_retrieve
    mr2ag_mllm_mode: str = field(
        default_factory=lambda: os.getenv("MR2AG_MLLM_MODE", "auto").strip().lower()
    )

    mr2ag_retrieval_top_k: int = field(
        default_factory=lambda: int(os.getenv("MR2AG_RETRIEVAL_TOP_K", "5"))
    )

    mr2ag_max_retrieval_depth: int = field(
        default_factory=lambda: int(os.getenv("MR2AG_MAX_RETRIEVAL_DEPTH", "5"))
    )

    mr2ag_min_relevance: float = field(
        default_factory=lambda: float(os.getenv("MR2AG_MIN_RELEVANCE", "0.45"))
    )

    mr2ag_retrieval_reflection_threshold: float = field(
        default_factory=lambda: float(os.getenv("MR2AG_RETRIEVAL_REFLECTION_THRESHOLD", "0.50"))
    )

    mr2ag_answer_confidence_threshold: float = field(
        default_factory=lambda: float(os.getenv("MR2AG_ANSWER_CONFIDENCE_THRESHOLD", "0.45"))
    )

    mr2ag_context_max_chars: int = field(
        default_factory=lambda: int(os.getenv("MR2AG_CONTEXT_MAX_CHARS", "6120"))
    )

    mr2ag_no_web_search: bool = field(
        default_factory=lambda: os.getenv("MR2AG_NO_WEB_SEARCH", "1").strip().lower() in {"1", "true", "yes", "on"}
    )
    
    # ===== GraphRAG + MLLM =====

    enable_graphrag_mllm: bool = field(
        default_factory=lambda: os.getenv("ENABLE_GRAPHRAG_MLLM", "0").strip().lower() in {"1", "true", "yes", "on"}
    )

    # local_global / local / global / drift / basic / proxy_only
    graphrag_mllm_mode: str = field(
        default_factory=lambda: os.getenv("GRAPHRAG_MLLM_MODE", "local_global").strip().lower()
    )

    # Microsoft GraphRAG workspace root. 该目录应已完成 graphrag init / graphrag index。
    graphrag_root: str = field(
        default_factory=lambda: os.getenv("GRAPHRAG_ROOT", "").strip()
    )

    # 可选：GraphRAG query 的 --data 参数；为空则使用 workspace 默认 output。
    graphrag_data_dir: str = field(
        default_factory=lambda: os.getenv("GRAPHRAG_DATA_DIR", "").strip()
    )

    graphrag_cli_bin: str = field(
        default_factory=lambda: os.getenv("GRAPHRAG_CLI_BIN", "graphrag").strip()
    )
    
    # GraphRAG CLI 可选专用 conda 环境。
    graphrag_conda_env: str = field(
        default_factory=lambda: os.getenv("GRAPHRAG_CONDA_ENV", "").strip()
    )

    graphrag_conda_exe: str = field(
        default_factory=lambda: os.getenv(
            "GRAPHRAG_CONDA_EXE",
            os.getenv("CONDA_EXE", "conda"),
        ).strip()
    )

    graphrag_query_methods: list[str] = field(
        default_factory=lambda: [
            x.strip().lower()
            for x in os.getenv("GRAPHRAG_QUERY_METHODS", "local,global").split(",")
            if x.strip()
        ]
    )

    graphrag_query_timeout_s: int = field(
        default_factory=lambda: int(os.getenv("GRAPHRAG_QUERY_TIMEOUT_S", "90"))
    )

    graphrag_visual_summary_max_chars: int = field(
        default_factory=lambda: int(os.getenv("GRAPHRAG_VISUAL_SUMMARY_MAX_CHARS", "1600"))
    )

    graphrag_context_max_chars: int = field(
        default_factory=lambda: int(os.getenv("GRAPHRAG_CONTEXT_MAX_CHARS", "6000"))
    )

    graphrag_retrieval_top_k: int = field(
        default_factory=lambda: int(os.getenv("GRAPHRAG_RETRIEVAL_TOP_K", "8"))
    )

    # 为 1 时，如果官方 GraphRAG 不可用就直接返回 fallback/error，不走 lexical fallback。
    graphrag_require_official: bool = field(
        default_factory=lambda: os.getenv("GRAPHRAG_REQUIRE_OFFICIAL", "0").strip().lower() in {"1", "true", "yes", "on"}
    )

    # 该分支结构上不调用 SearchTool/WebpageTool；此开关只用于 trace 和 bash 显式约束。
    graphrag_no_web_search: bool = field(
        default_factory=lambda: os.getenv("GRAPHRAG_NO_WEB_SEARCH", "1").strip().lower() in {"1", "true", "yes", "on"}
    )
    
    # ===== VLLM IO Debug =====
    # 是否打印每次发给 VLLM 的 messages 和返回结果
    debug_vllm_io: bool = field(
        default_factory=lambda: os.getenv("DEBUG_VLLM_IO", "0").strip().lower() in {"1", "true", "yes", "on"}
    )

    # 打印时单条文本最多保留多少字符，避免日志爆炸
    debug_vllm_max_chars: int = field(
        default_factory=lambda: int(os.getenv("DEBUG_VLLM_MAX_CHARS", "4000"))
    )

    # 是否把 messages 原样完整打印；默认 False，走裁剪版
    debug_vllm_full_messages: bool = field(
        default_factory=lambda: os.getenv("DEBUG_VLLM_FULL_MESSAGES", "0").strip().lower() in {"1", "true", "yes", "on"}
    )
    
    kg_retrieval_backend: str = field(default_factory=lambda: os.getenv("KG_RETRIEVAL_BACKEND", "auto"))
    qianfan_api_key: str = field(default_factory=lambda: os.getenv("QIANFAN_API_KEY", "bce-v3/ALTAK-"))
    qianfan_kb_ids: list[str] = field(
        default_factory=lambda: [x.strip() for x in os.getenv("QIANFAN_KB_IDS", "ecc0fa40-3516-4b29-a1d4-b2006bbf862a").split(",") if x.strip()]
    )

    dbpedia_endpoint: str = field(default_factory=lambda: os.getenv("DBPEDIA_ENDPOINT", "https://dbpedia.org/sparql"))
    dbpedia_timeout: int = field(default_factory=lambda: int(os.getenv("DBPEDIA_TIMEOUT", "10")))
    dbpedia_top_k: int = field(default_factory=lambda: int(os.getenv("DBPEDIA_TOP_K", "5")))

    # ===== Retrieval annotation docs =====
    include_annotation_retrieval_docs: bool = field(
        default_factory=lambda: os.getenv("INCLUDE_ANNOTATION_RETRIEVAL_DOCS", "0").strip().lower() in {"1", "true", "yes", "on"}
    )
    annotation_retrieval_datasets: list[str] = field(
        default_factory=lambda: [x.strip().lower() for x in os.getenv(
            "ANNOTATION_RETRIEVAL_DATASETS",
            "m3cot,aokvqa,cmmqa,scienceqa,mmmu_pro"
        ).split(",") if x.strip()]
    )
    
    # ===== Visual-semantic web search =====
    enable_visual_semantic_web_search: bool = field(
        default_factory=lambda: os.getenv("ENABLE_VISUAL_SEMANTIC_WEB_SEARCH", "1").strip().lower() in {"1", "true", "yes", "on"}
    )
    web_search_match_threshold: float = field(
        default_factory=lambda: float(os.getenv("WEB_SEARCH_MATCH_THRESHOLD", "0.55"))
    )
    web_search_max_rounds: int = field(
        default_factory=lambda: int(os.getenv("WEB_SEARCH_MAX_ROUNDS", "5"))
    )
    
    # ===== Web search tool switch for KG ablation experiments =====
    # 控制 agentic KG-RAG 路线中是否允许使用 SearchTool/WebpageTool 相关工具：
    # search_web / open_webpage / extract_images。
    enable_web_search_tool: bool = field(
        default_factory=lambda: os.getenv("ENABLE_WEB_SEARCH_TOOL", "1").strip().lower()
        in {"1", "true", "yes", "on"}
    )

    # 默认对 kg_difficulty / kg_only 支持 no-web ablation。
    # 如果 CLI 没有显式传 --enable-web-search-tool / --disable-web-search-tool，
    # run_eval 会根据该列表和 ablation_profile 自动决定是否关闭 web search tool。
    no_web_search_ablation_profiles: list[str] = field(
        default_factory=lambda: [
            x.strip().lower()
            for x in os.getenv(
                "NO_WEB_SEARCH_ABLATION_PROFILES",
                "kg_difficulty,kg_only,kg_only1",
            ).split(",")
            if x.strip()
        ]
    )

    # 0：只禁用 search_web/open_webpage/extract_images，保留原有 KG backend 逻辑；
    # 1：当 web search tool 关闭时，也禁止 auto/hybrid/qianfan/dbpedia 等在线外部 KG fallback，
    #    强制回到 light_rag/proxy_graph 的本地文档代理图谱检索。
    strict_no_external_kg_when_web_disabled: bool = field(
        default_factory=lambda: os.getenv("STRICT_NO_EXTERNAL_KG_WHEN_WEB_DISABLED", "0").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    
    
    # ===== External LLM routing: DashScope / XIAOAI.PLUS =====

    dashscope_api_base: str = field(
        default_factory=lambda: os.getenv(
            "DASHSCOPE_API_BASE",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    )
    dashscope_api_key: str = field(
        default_factory=lambda: os.getenv("DASHSCOPE_API_KEY", "sk-")
    )
    dashscope_text_model: str = field(
        default_factory=lambda: os.getenv("DASHSCOPE_TEXT_MODEL", "qwen3-vl-8b-thinking")
    )
    dashscope_vision_model: str = field(
        default_factory=lambda: os.getenv("DASHSCOPE_VISION_MODEL", "qwen3-vl-32b-instruct")
    )

    xiaoai_api_base: str = field(
        default_factory=lambda: os.getenv(
            "XIAOAI_API_BASE",
            os.getenv("GPT_RETRIEVAL_SEED_API_BASE", "https://xiaoai.plus/v1"),
        )
    )
    xiaoai_api_key: str = field(
        default_factory=lambda: os.getenv(
            "XIAOAI_API_KEY",
            os.getenv("GPT_RETRIEVAL_SEED_API_KEY", "sk-"),
        )
    )
    xiaoai_text_model: str = field(
        default_factory=lambda: os.getenv(
            "XIAOAI_TEXT_MODEL",
            os.getenv("GPT_RETRIEVAL_SEED_MODEL", "gpt-5.1-chat"),
        )
    )
    xiaoai_vision_model: str = field(
        default_factory=lambda: os.getenv(
            "XIAOAI_VISION_MODEL",
            os.getenv("XIAOAI_TEXT_MODEL", os.getenv("GPT_RETRIEVAL_SEED_MODEL", "gpt-5.1-chat")),
        )
    )

    # 显式后端开关，优先级最高。
    # 可选值：local / vllm / dashscope / xiaoai / xiaoai_plus / xiaoai.plus
    # 留空时走下面的旧 bool 开关，保持兼容。
    planner_backend: str = field(
        default_factory=lambda: os.getenv("PLANNER_BACKEND", "").strip().lower()
    )
    reasoning_backend: str = field(
        default_factory=lambda: os.getenv("REASONING_BACKEND", "").strip().lower()
    )
    vision_backend: str = field(
        default_factory=lambda: os.getenv("VISION_BACKEND", "").strip().lower()
    )

    # 当 *_BACKEND 留空时，*_USE_XIAOAI=1 优先于 *_USE_DASHSCOPE=1。
    planner_use_xiaoai: bool = field(
        default_factory=lambda: os.getenv("PLANNER_USE_XIAOAI", "0").strip().lower() in {"1", "true", "yes", "on"}
    )
    reasoning_use_xiaoai: bool = field(
        default_factory=lambda: os.getenv("REASONING_USE_XIAOAI", "0").strip().lower() in {"1", "true", "yes", "on"}
    )
    vision_use_xiaoai: bool = field(
        default_factory=lambda: os.getenv("VISION_USE_XIAOAI", "0").strip().lower() in {"1", "true", "yes", "on"}
    )

    # DashScope 开关继续保留。
    planner_use_dashscope: bool = field(
        default_factory=lambda: os.getenv("PLANNER_USE_DASHSCOPE", "1").strip().lower() in {"1", "true", "yes", "on"}
    )
    reasoning_use_dashscope: bool = field(
        default_factory=lambda: os.getenv("REASONING_USE_DASHSCOPE", "0").strip().lower() in {"1", "true", "yes", "on"}
    )
    vision_use_dashscope: bool = field(
        default_factory=lambda: os.getenv("VISION_USE_DASHSCOPE", "1").strip().lower() in {"1", "true", "yes", "on"}
    )

    dbpedia_generic_entity_blocklist: list[str] = field(
        default_factory=lambda: [x.strip().lower() for x in os.getenv(
            "DBPEDIA_GENERIC_ENTITY_BLOCKLIST",
            "person,people,human,humans,place,location,area,thing,object,entity,scene,setting,view,vehicle,car,building,street,urban,rural,residential,commercial,private,public"
        ).split(",") if x.strip()]
    )
    
    max_subquestion_rewrites: int = field(
        default_factory=lambda: int(os.getenv("MAX_SUBQUESTION_REWRITES", "3"))
    )

    min_subquestion_grounding_score: float = field(
        default_factory=lambda: float(os.getenv("MIN_SUBQUESTION_GROUNDING_SCORE", "0.45"))
    )


    final_answer_confidence_threshold: float = field(
        default_factory=lambda: float(os.getenv("FINAL_ANSWER_CONFIDENCE_THRESHOLD", "0.60"))
    )

    enable_search_domain_filter: bool = field(
        default_factory=lambda: os.getenv("ENABLE_SEARCH_DOMAIN_FILTER", "1").strip().lower() in {"1", "true", "yes", "on"}
    )
    
    
    min_visual_grounding_signal: float = field(
        default_factory=lambda: float(os.getenv("MIN_VISUAL_GROUNDING_SIGNAL", "0.40"))
    )

    local_visual_bucket_cap: int = field(
        default_factory=lambda: int(os.getenv("LOCAL_VISUAL_BUCKET_CAP", "4"))
    )

    graph_signal_min_confidence: float = field(
        default_factory=lambda: float(os.getenv("GRAPH_SIGNAL_MIN_CONFIDENCE", "0.55"))
    )

    graph_signal_min_query_results: int = field(
        default_factory=lambda: int(os.getenv("GRAPH_SIGNAL_MIN_QUERY_RESULTS", "2"))
    )

    graph_signal_min_kb_score: float = field(
        default_factory=lambda: float(os.getenv("GRAPH_SIGNAL_MIN_KB_SCORE", "0.72"))
    )
    
    # ===== GPT-5.1 retrieval seed via XIAOAI.PLUS =====
    enable_gpt_retrieval_seed: bool = field(
        default_factory=lambda: os.getenv("ENABLE_GPT_RETRIEVAL_SEED", "1").strip().lower() in {"1", "true", "yes", "on"}
    )
    gpt_retrieval_seed_api_base: str = field(
        
        default_factory=lambda: os.getenv("GPT_RETRIEVAL_SEED_API_BASE", "https://xiaoai.plus/v1")
    )
    gpt_retrieval_seed_api_key: str = field(
        default_factory=lambda: os.getenv("GPT_RETRIEVAL_SEED_API_KEY", "sk-")
    )
    gpt_retrieval_seed_model: str = field(
        default_factory=lambda: os.getenv("GPT_RETRIEVAL_SEED_MODEL", "gpt-5.1-chat")
    )
    gpt_retrieval_seed_timeout_s: float = field(
        default_factory=lambda: float(os.getenv("GPT_RETRIEVAL_SEED_TIMEOUT_S", "60"))
    )
    gpt_retrieval_seed_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("GPT_RETRIEVAL_SEED_MAX_TOKENS", "1200"))
    )
    gpt_retrieval_seed_max_docs: int = field(
        default_factory=lambda: int(os.getenv("GPT_RETRIEVAL_SEED_MAX_DOCS", "6"))
    )
    gpt_retrieval_seed_include_image: bool = field(
        default_factory=lambda: os.getenv("GPT_RETRIEVAL_SEED_INCLUDE_IMAGE", "1").strip().lower() in {"1", "true", "yes", "on"}
    )

    # ===== Route-specific max tokens =====
    planner_api_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("PLANNER_API_MAX_TOKENS", "384"))
    )
    reasoning_api_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("REASONING_API_MAX_TOKENS", "640"))
    )
    vision_api_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("VISION_API_MAX_TOKENS", "768"))
    )
    verify_api_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("VERIFY_API_MAX_TOKENS", "384"))
    )
    synthesize_api_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("SYNTHESIZE_API_MAX_TOKENS", "640"))
    )
    
    enable_baseline_short_circuit: bool = field(
        default_factory=lambda: os.getenv("ENABLE_BASELINE_SHORT_CIRCUIT", "1").strip().lower() in {"1", "true", "yes", "on"}
    )

    enable_fallback_on_abstain: bool = field(
        default_factory=lambda: os.getenv("ENABLE_FALLBACK_ON_ABSTAIN", "1").strip().lower() in {"1", "true", "yes", "on"}
    )


settings = Settings()
