from __future__ import annotations

from collections import Counter


def accuracy(predictions: list[str], references: list[str]) -> float:
    if not references:
        return 0.0
    return sum(p == r for p, r in zip(predictions, references)) / len(references)


def _ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1)))


def bleu_1_to_4(predictions: list[str], references: list[str]) -> float:
    if not predictions:
        return 0.0
    import math

    precisions = []
    for n in range(1, 5):
        overlap = total = 0
        for pred, ref in zip(predictions, references):
            pred_counts = _ngrams(pred.split(), n)
            ref_counts = _ngrams(ref.split(), n)
            overlap += sum(min(count, ref_counts[gram]) for gram, count in pred_counts.items())
            total += sum(pred_counts.values())
        precisions.append((overlap + 1.0) / (total + 1.0))
    pred_len = sum(len(x.split()) for x in predictions)
    ref_len = sum(len(x.split()) for x in references)
    bp = 1.0 if pred_len > ref_len else math.exp(1.0 - ref_len / max(1, pred_len))
    return float(bp * math.exp(sum(math.log(p) for p in precisions) / 4.0))


def rouge_l(predictions: list[str], references: list[str]) -> float:
    def lcs(a: list[str], b: list[str]) -> int:
        prev = [0] * (len(b) + 1)
        for tok_a in a:
            cur = [0]
            for j, tok_b in enumerate(b, start=1):
                cur.append(prev[j - 1] + 1 if tok_a == tok_b else max(prev[j], cur[-1]))
            prev = cur
        return prev[-1]

    scores = []
    for pred, ref in zip(predictions, references):
        p, r = pred.split(), ref.split()
        if not p or not r:
            scores.append(0.0)
            continue
        common = lcs(p, r)
        precision = common / len(p)
        recall = common / len(r)
        scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return sum(scores) / max(1, len(scores))

