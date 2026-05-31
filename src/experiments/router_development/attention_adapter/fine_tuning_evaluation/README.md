# Downstream GLUE-Style Adapter Evaluation Suite

This directory contains the downstream GLUE-style evaluation suite for decoder-only causal language models. Classification examples are converted into prompts, and predictions are produced by scoring task-specific verbalizer candidates.

Task-specific runbooks live in:

```text
task_readmes/sst2.md    SST-2 fine-tuning and submission commands
task_readmes/rte.md     RTE fine-tuning and submission commands
```

The suite currently supports:

```text
sst2    binary sentiment classification from GLUE SST-2
rte     binary textual entailment from GLUE RTE
```

Run commands from the project root with `PYTHONPATH=src`.

