from dataclasses import dataclass


@dataclass
class CVConfig:
    project_name: str = "drt_sae_cv_clip_vit_b16_l9"
    model_name: str = "openai/clip-vit-base-patch16"
    image_dataset_root: str = "/mnt/hdd/coco2017/train2017"
    caption_dataset_name: str = "HuggingFaceM4/COCO"
    caption_dataset_split: str = "train"
    image_column: str = "image"
    default_layer_id: int = 9
    recommended_layer_ids: tuple[int, ...] = (7, 9, 10)
    image_size: int = 224
    activation_cache_dir: str = "artifacts/activations/cv_clip_vit_b16_l9"
    concept_cache_dir: str = "artifacts/concepts/cv_clip_vit_b16_l9"
    teacher_checkpoint: str = "artifacts/checkpoints/cv_teacher_l9.pt"
    router_checkpoint: str = "artifacts/checkpoints/cv_drt_sae_l9.pt"
    shard_tokens: int = 262144
    cache_image_batch_size: int = 64
    cache_image_limit: int | None = 25000
    cache_time_limit_hours: float | None = 5.0
    train_batch_tokens: int = 32768
    teacher_steps: int = 15000
    router_steps: int = 15000
    d_model: int = 768
    n_latents_factor: int = 16
    teacher_k: int = 32
    router_k: int = 16
    router_dim: int = 64
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


cfg = CVConfig()
