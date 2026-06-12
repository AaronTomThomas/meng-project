import csv
import json
from pathlib import Path
from statistics import mean, stdev

def fmt_mean_std(xs, digits=6, suffix=""):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "-"
    if len(xs) == 1:
        return f"{xs[0]:.{digits}f}{suffix}"
    return f"{mean(xs):.{digits}f} ± {stdev(xs):.{digits}f}{suffix}"

def compact_layers(layer_indices):
    if layer_indices is None:
        return "-"
    if isinstance(layer_indices, str):
        parts = [int(x) for x in layer_indices.split(",") if x.strip()]
    else:
        parts = [int(x) for x in layer_indices]

    if not parts:
        return "-"

    parts = sorted(parts)
    if parts == list(range(parts[0], parts[-1] + 1)):
        return f"{parts[0]}-{parts[-1]}"
    return ",".join(map(str, parts))

def dataset_label(name, cfg):
    if name.startswith("wt2_"):
        return "WikiText-2"
    if name.startswith("ptb_"):
        return "PTB"
    if name.startswith("owt10k_"):
        return "OpenWebText-10k"

    dn = cfg.get("dataset_name", "")
    dc = cfg.get("dataset_config", "")

    if dn == "wikitext" and dc == "wikitext-2-raw-v1":
        return "WikiText-2"
    if "ptb" in dn:
        return "PTB"
    if "openwebtext" in dn:
        return "OpenWebText-10k"

    return dc or dn or "unknown"

def model_label(name, cfg):
    model = cfg.get("model_name", "")

    if "pythia-1b" in name or "pythia-1b" in model:
        return "Pythia-1B"
    if "pythia160m" in name or "pythia-160m" in model:
        return "Pythia-160M"
    if "gpt2" in name or model.endswith("gpt2") or model == "openai-community/gpt2":
        return "GPT-2 small"

    return model or "unknown"

def method_label(name, s, cfg):
    method = s.get("method") or cfg.get("method") or "unknown"

    if method == "akaza_freez":
        b = cfg.get("bottleneck_dim")
        return f"AKAZA-FreeZ b{b}" if b is not None else "AKAZA-FreeZ"

    if method == "lora":
        target = s.get("peft_target_profile") or cfg.get("peft_target_profile") or "?"
        r = cfg.get("lora_rank")
        a = cfg.get("lora_alpha")
        return f"LoRA {target} r{r} a{a}"

    if method == "loreft":
        r = cfg.get("reft_rank")
        drop = cfg.get("reft_dropout")
        return f"LoReFT r{r} drop{drop}"

    return method

def safe_get(summary, key):
    return summary.get(key, None)

rows = []

for path in sorted(Path(".").glob("*.summary.json")):
    with open(path) as f:
        s = json.load(f)

    cfg = s.get("config", {})
    name = path.name.replace(".pt.summary.json", "")

    row = {
        "run": name,
        "model": model_label(name, cfg),
        "dataset": dataset_label(name, cfg),
        "method": method_label(name, s, cfg),
        "raw_method": s.get("method") or cfg.get("method"),
        "seed": cfg.get("seed"),
        "params": s.get("num_trainable_params"),
        "layers": compact_layers(s.get("layer_indices") or cfg.get("layer_indices")),
        "block_size": cfg.get("block_size"),
        "batch_size": cfg.get("batch_size"),
        "train_chunks": s.get("train_chunks") or cfg.get("max_train_chunks"),
        "val_chunks": s.get("val_chunks") or cfg.get("max_val_chunks"),
        "test_chunks": s.get("test_chunks") or cfg.get("max_test_chunks"),
        "lr": cfg.get("lr"),
        "best_epoch": safe_get(s, "best_epoch"),
        "baseline_val_loss": safe_get(s, "baseline_val_loss"),
        "best_val_loss": safe_get(s, "best_val_loss"),
        "val_improvement": safe_get(s, "best_val_improvement_nats_per_token"),
        "baseline_test_loss": safe_get(s, "baseline_test_loss"),
        "best_test_loss": safe_get(s, "best_test_loss"),
        "test_improvement": safe_get(s, "best_test_improvement_nats_per_token"),
        "test_ppl_reduction_pct": 100.0 * safe_get(s, "best_test_relative_ppl_reduction"),
    }
    rows.append(row)

if not rows:
    raise SystemExit("No *.summary.json files found.")

rows.sort(key=lambda r: (
    r["model"],
    r["dataset"],
    r["method"],
    r["block_size"] or -1,
    r["train_chunks"] or -1,
    r["seed"] if r["seed"] is not None else -1,
))

# Per-run CSV
run_fields = list(rows[0].keys())
with open("peft_eval_collated_runs_v2.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=run_fields)
    writer.writeheader()
    writer.writerows(rows)

# Grouped results
groups = {}
for r in rows:
    key = (
        r["model"],
        r["dataset"],
        r["method"],
        r["params"],
        r["layers"],
        r["block_size"],
        r["train_chunks"],
        r["val_chunks"],
        r["test_chunks"],
        r["lr"],
    )
    groups.setdefault(key, []).append(r)

group_rows = []
for key, rs in sorted(groups.items()):
    model, dataset, method, params, layers, block, train_chunks, val_chunks, test_chunks, lr = key

    group_rows.append({
        "model": model,
        "dataset": dataset,
        "method": method,
        "params": params,
        "n": len(rs),
        "seeds": ",".join(str(r["seed"]) for r in rs),
        "layers": layers,
        "block_size": block,
        "train_chunks": train_chunks,
        "val_chunks": val_chunks,
        "test_chunks": test_chunks,
        "lr": lr,
        "val_improvement_mean": mean(r["val_improvement"] for r in rs),
        "val_improvement_std": stdev(r["val_improvement"] for r in rs) if len(rs) > 1 else 0.0,
        "test_improvement_mean": mean(r["test_improvement"] for r in rs),
        "test_improvement_std": stdev(r["test_improvement"] for r in rs) if len(rs) > 1 else 0.0,
        "test_ppl_reduction_pct_mean": mean(r["test_ppl_reduction_pct"] for r in rs),
        "test_ppl_reduction_pct_std": stdev(r["test_ppl_reduction_pct"] for r in rs) if len(rs) > 1 else 0.0,
    })

group_fields = list(group_rows[0].keys())
with open("peft_eval_collated_groups_v2.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=group_fields)
    writer.writeheader()
    writer.writerows(group_rows)

# Markdown
lines = []
lines.append("# PEFT evaluation collated results\n")

lines.append("## Grouped thesis table\n")
lines.append("| Model | Dataset | Method | Params | n | Seeds | Layers | Block | Chunks | LR | Val imp | Test imp | Test PPL reduction |")
lines.append("|---|---|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|")

for g in group_rows:
    chunks = f"{g['train_chunks']}/{g['val_chunks']}/{g['test_chunks']}"
    vals = [r["val_improvement"] for r in groups[(
        g["model"], g["dataset"], g["method"], g["params"], g["layers"],
        g["block_size"], g["train_chunks"], g["val_chunks"], g["test_chunks"], g["lr"]
    )]]
    tests = [r["test_improvement"] for r in groups[(
        g["model"], g["dataset"], g["method"], g["params"], g["layers"],
        g["block_size"], g["train_chunks"], g["val_chunks"], g["test_chunks"], g["lr"]
    )]]
    ppls = [r["test_ppl_reduction_pct"] for r in groups[(
        g["model"], g["dataset"], g["method"], g["params"], g["layers"],
        g["block_size"], g["train_chunks"], g["val_chunks"], g["test_chunks"], g["lr"]
    )]]

    lines.append(
        f"| {g['model']} | {g['dataset']} | {g['method']} "
        f"| {g['params']} | {g['n']} | {g['seeds']} | {g['layers']} "
        f"| {g['block_size']} | {chunks} | {g['lr']} "
        f"| {fmt_mean_std(vals)} "
        f"| {fmt_mean_std(tests)} "
        f"| {fmt_mean_std(ppls, digits=3, suffix='%')} |"
    )

lines.append("\n## Per-run table\n")
lines.append("| Model | Dataset | Run | Method | Seed | Params | Layers | Block | Chunks | LR | Best epoch | Val imp | Test imp | Test PPL reduction |")
lines.append("|---|---|---|---|---:|---:|---|---:|---|---:|---:|---:|---:|---:|")

for r in rows:
    chunks = f"{r['train_chunks']}/{r['val_chunks']}/{r['test_chunks']}"
    lines.append(
        f"| {r['model']} | {r['dataset']} | `{r['run']}` | {r['method']} "
        f"| {r['seed']} | {r['params']} | {r['layers']} | {r['block_size']} "
        f"| {chunks} | {r['lr']} | {r['best_epoch']} "
        f"| {r['val_improvement']:.6f} "
        f"| {r['test_improvement']:.6f} "
        f"| {r['test_ppl_reduction_pct']:.3f}% |"
    )

Path("peft_eval_collated_v2.md").write_text("\n".join(lines))

print("Wrote:")
print("  peft_eval_collated_runs_v2.csv")
print("  peft_eval_collated_groups_v2.csv")
print("  peft_eval_collated_v2.md")
print()
print(Path("peft_eval_collated_v2.md").read_text())
