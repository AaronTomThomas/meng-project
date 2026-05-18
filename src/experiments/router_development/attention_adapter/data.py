from dataclasses import dataclass, replace
import hashlib
from pathlib import Path

from experiments.router_development.attention_adapter.config import AdapterFineTuneConfig
from datasets import get_dataset_split_names, load_dataset
import torch


def short_hash(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()[:10]


def _text_cache_key(cfg: AdapterFineTuneConfig, text_field: str) -> str:
    return (
        f"{cfg.model_name}|{cfg.dataset_name}|{cfg.dataset_config}|{cfg.dataset_revision}|{cfg.split}|"
        f"text={text_field}|max_texts={cfg.max_texts}|block={cfg.block_size}|"
        f"max_chunks={cfg.max_chunks}"
    )


def _official_split_names(cfg: AdapterFineTuneConfig) -> list[str]:
    kwargs = {}
    if cfg.dataset_revision:
        kwargs["revision"] = cfg.dataset_revision
    if cfg.dataset_config:
        return get_dataset_split_names(cfg.dataset_name, cfg.dataset_config, **kwargs)
    return get_dataset_split_names(cfg.dataset_name, **kwargs)


def _base_split_names(split_expr: str) -> list[str]:
    """
    Extract official split names from Hugging Face split expressions.

    Examples:
      train             -> ["train"]
      train[:80%]       -> ["train"]
      train[80%:90%]    -> ["train"]
      train+validation  -> ["train", "validation"]
    """
    bases: list[str] = []
    for part in split_expr.split("+"):
        base = part.split("[", 1)[0].strip()
        if base:
            bases.append(base)
    return bases or [split_expr]


def validate_official_splits(cfg: AdapterFineTuneConfig) -> tuple[list[str], bool]:
    requested = {
        "train_split": cfg.train_split,
        "val_split": cfg.val_split,
        "test_split": cfg.test_split,
    }

    try:
        split_names = _official_split_names(cfg)
    except Exception as exc:
        requested_splits = sorted(set(requested.values()))
        print(
            "[data] could not inspect official dataset split metadata before loading; "
            f"will request named splits directly: {requested_splits}. Reason: {exc}"
        )
        return requested_splits, False

    missing: dict[str, str] = {}
    for name, split_expr in requested.items():
        for base in _base_split_names(split_expr):
            if base not in split_names:
                missing[name] = split_expr
                break

    if missing:
        details = ", ".join(f"{name}={split!r}" for name, split in missing.items())
        raise ValueError(
            f"Requested split(s) are not official splits for "
            f"{cfg.dataset_name}/{cfg.dataset_config}: {details}. "
            f"Available official splits: {split_names}"
        )

    print(f"[data] official dataset splits found: {split_names}")
    return split_names, True

# def validate_official_splits(cfg: AdapterFineTuneConfig) -> tuple[list[str], bool]:
#     requested = {
#         "train_split": cfg.train_split,
#         "val_split": cfg.val_split,
#         "test_split": cfg.test_split,
#     }
#     try:
#         split_names = _official_split_names(cfg)
#     except Exception as exc:
#         requested_splits = sorted(set(requested.values()))
#         print(
#             "[data] could not inspect official dataset split metadata before loading; "
#             f"will request named splits directly: {requested_splits}. Reason: {exc}"
#         )
#         return requested_splits, False

#     missing = {name: split for name, split in requested.items() if split not in split_names}
#     if missing:
#         details = ", ".join(f"{name}={split!r}" for name, split in missing.items())
#         raise ValueError(
#             f"Requested split(s) are not official splits for "
#             f"{cfg.dataset_name}/{cfg.dataset_config}: {details}. "
#             f"Available official splits: {split_names}"
#         )
#     print(f"[data] official dataset splits found: {split_names}")
#     return split_names, True


def load_and_pack_texts(
    cfg: AdapterFineTuneConfig,
    tokenizer,
    text_field: str | None = None,
) -> torch.Tensor:
    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    text_field = text_field or getattr(cfg, "text_field", "text")

    cache_path = cache_dir / f"{short_hash(_text_cache_key(cfg, text_field))}__chunks.pt"
    if cache_path.exists():
        print(f"[cache] loading chunks from {cache_path}")
        return torch.load(cache_path)
    print("[data] loading dataset...")
    kwargs = {}
    if cfg.dataset_revision:
        kwargs["revision"] = cfg.dataset_revision
    if cfg.dataset_config:
        ds = load_dataset(cfg.dataset_name, cfg.dataset_config, split=cfg.split, **kwargs)
    else:
        ds = load_dataset(cfg.dataset_name, split=cfg.split, **kwargs)

    token_blocks = []
    total_texts = 0
    print("[data] tokenizing and packing texts...")
    for ex in ds:
        text = ex[text_field]
        if not isinstance(text, str) or not text.strip():
            continue
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        if ids.numel() < cfg.block_size:
            continue
        n_blocks = ids.numel() // cfg.block_size
        ids = ids[: n_blocks * cfg.block_size].view(n_blocks, cfg.block_size)
        token_blocks.append(ids)
        total_texts += 1
        if total_texts >= cfg.max_texts:
            break
    if not token_blocks:
        raise ValueError("No usable token blocks found.")
    chunks = torch.cat(token_blocks, dim=0)[: cfg.max_chunks]
    print(f"[data] packed {chunks.shape[0]} chunks from {total_texts} texts")
    torch.save(chunks.cpu(), cache_path)
    return chunks

def load_chunks_for_split(
    cfg: AdapterFineTuneConfig,
    tokenizer,
    *,
    split: str,
    max_texts: int,
    max_chunks: int,
) -> torch.Tensor:
    split_cfg = replace(cfg, split=split, max_texts=max_texts, max_chunks=max_chunks)
    chunks = load_and_pack_texts(split_cfg, tokenizer, text_field=cfg.text_field).cpu()
    if max_chunks > 0:
        chunks = chunks[:max_chunks]
    return chunks

@dataclass(frozen=True)
class AdapterFineTuneData:
    train: torch.Tensor
    val: torch.Tensor
    test: torch.Tensor
    official_splits: list[str]
    official_splits_checked: bool

    @property
    def block_size(self) -> int:
        return int(self.train.shape[1])


def load_adapter_finetune_data(cfg: AdapterFineTuneConfig, tokenizer) -> AdapterFineTuneData:
    official_splits, official_splits_checked = validate_official_splits(cfg)
    if official_splits_checked:
        print(f"[data] official dataset splits: {official_splits}")

    print("[data] loading official train/val/test chunks separately")
    data = AdapterFineTuneData(
        train=load_chunks_for_split(
            cfg,
            tokenizer,
            split=cfg.train_split,
            max_texts=cfg.max_train_texts,
            max_chunks=cfg.max_train_chunks,
        ),
        val=load_chunks_for_split(
            cfg,
            tokenizer,
            split=cfg.val_split,
            max_texts=cfg.max_val_texts,
            max_chunks=cfg.max_val_chunks,
        ),
        test=load_chunks_for_split(
            cfg,
            tokenizer,
            split=cfg.test_split,
            max_texts=cfg.max_test_texts,
            max_chunks=cfg.max_test_chunks,
        ),
        official_splits=official_splits,
        official_splits_checked=official_splits_checked,
    )
    print(f"[data] train chunks={data.train.shape[0]} block_size={data.block_size}")
    print(f"[data] val   chunks={data.val.shape[0]} block_size={data.val.shape[1]}")
    print(f"[data] test  chunks={data.test.shape[0]} block_size={data.test.shape[1]}")
    return data
