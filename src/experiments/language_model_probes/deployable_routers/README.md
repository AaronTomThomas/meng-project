# Deployable Routers

This directory contains the deployable-router code that remains thesis-relevant
after pruning the historical router attempts.

## Current Evidence

The main result is the gradient-proxy router robustness check:

- layer 4: mean gain vs `soft` = `0.002978`, positive `3/3` seeds
- layer 4 best seed: mean cost `3.697862`, gain `0.004342`, switch rate `0.1465`
- layer 11: mean gain vs `soft` = `-0.000208`, positive `0/3` seeds

The key artifact is:

```text
outputs/deployable_routers/gradient_proxy_router/layer_robustness_l4_l11/summary.json
```

## Interpretation

The honest thesis claim is narrow:

- cheap/local router families failed to close the oracle gap robustly
- gradient-proxy supervision exposes a small deployable signal at layer 4
- that signal is layer-dependent and does not transfer cleanly to layer 11

This should not be presented as a solved routing problem.

## Related Weak Baselines

The sparse deployable switch result is useful as a weak positive baseline:

- mean cost `3.700881`
- gain vs `soft` = `0.001323`
- switch rate `0.0151`

The selective pairwise follow-up is weaker:

- mean cost around `3.701288`
- gain vs `soft` around `0.000916`

These are supporting appendix results, not the primary deployable-router claim.

## Code

- `gradient_proxy_router/datasets.py`: builds and augments deployable datasets.
- `gradient_proxy_router/router.py`: trains, evaluates, and runs robustness checks.
- `gradient_proxy_router/cli.py`: single command facade for all runnable workflows.
- `gradient_proxy_router/utils.py`: shared low-level utilities.
