from dataclasses import dataclass


@dataclass
class ModelConfig:
    block_size: int = 8  # maximum context length
    vocab_size: int = 74
    n_head: int = 4
    n_embd: int = 32
    n_layer: int = 3
    dropout: float = 0.2


@dataclass
class TrainerConfig:
    batch_size: int = 32  # how many independent sequences processed in parallel?
    max_iters: int = 5000
    eval_interval: int = 500
    learning_rate: float = 3e-4
    eval_iters: int = 200
    checkpoint: str = ""
    save_interval: int = 0


@dataclass
class MLFlowConfig:
    experiment: str = "sail_experiment"
    log_interval: int = 10
    deep_log_interval: int = 100
    system_metrics: bool = False
