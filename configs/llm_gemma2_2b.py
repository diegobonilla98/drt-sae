from dataclasses import dataclass


@dataclass
class LLMConfig:
    project_name: str = "drt_sae_llm_gemma2_2b_l13"
    model_name: str = "google/gemma-2-2b"
    fallback_model_name: str = "EleutherAI/pythia-410m-deduped"
    dataset_name: str = "Skylion007/openwebtext"
    dataset_config: str | None = None
    dataset_split: str = "train"
    text_column: str = "text"
    default_layer_id: int = 13
    recommended_layer_ids: tuple[int, ...] = (9, 13, 17)
    max_length: int = 1024
    activation_cache_dir: str = "artifacts/activations/llm_gemma2_2b_l13"
    concept_cache_dir: str = "artifacts/concepts/llm_gemma2_2b_l13"
    teacher_checkpoint: str = "artifacts/checkpoints/llm_teacher_l13.pt"
    router_checkpoint: str = "artifacts/checkpoints/llm_drt_sae_l13.pt"
    shard_tokens: int = 262144
    cache_text_batch_size: int = 8
    train_batch_tokens: int = 8192
    teacher_steps: int = 20000
    router_steps: int = 20000
    d_model: int | None = None
    n_latents_factor: int = 16
    teacher_k: int = 64
    router_k: int = 32
    router_dim: int = 128
    score_chunk_size: int = 8192
    route_negative_count: int = 128
    lr_teacher: float = 3e-4
    lr_router: float = 3e-4
    lr_decoder: float = 1e-4
    weight_decay: float = 0.0
    coef_loss_weight: float = 1e-5
    route_loss_weight: float = 0.2
    load_loss_weight: float = 1e-2
    router_bias_rate: float = 1e-3
    log_every: int = 50
    save_every: int = 1000
    dtype: str = "bfloat16"
    device: str = "cuda"
    concept_top_n: int = 32
    concept_near_n: int = 16
    concept_low_n: int = 16
    concept_min_examples: int = 8
    concept_llm_model: str = "gpt-5-mini"
    openai_env_key: str = "OPENAI_API_KEY"


cfg = LLMConfig()
