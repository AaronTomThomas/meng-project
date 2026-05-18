from experiments.gpt2_probe_utils import (
    continue_from_modified_block_gpt2,
    extract_head_qkv_and_teacher_outputs_gpt2,
    head_slice,
    load_and_pack_texts,
    mean_next_token_nll,
    parse_head_indices,
    run_to_block_and_cache_tensors,
)
from experiments.attention_learners import LearnerHyperParams
from experiments.language_model_probes.probe_utils import LearnerRegistry


BASE_LEARNERS = ["soft", "sharp", "window_soft", "weighted_linear"]
LEARNER_REGISTRY = LearnerRegistry(BASE_LEARNERS)