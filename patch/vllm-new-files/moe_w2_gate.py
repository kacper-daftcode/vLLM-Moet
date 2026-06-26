# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Confidence-gated FP4 re-forward for the 2-bit MoE path (directive 2 / Step B).

When the 2-bit base emits a LOW-CONFIDENCE decode token, this gate re-runs the
step with the token's routed experts pulled up to FP4 (via the delta tier's
`force_promote`) and re-decides. Offline validation on a coding corpus
(gate_validate.py) showed that gating on `max_prob <= 0.67` (~30% of tokens)
recovers ~90% of the 2-bit->FP4 top-1 agreement gap and ~61% of the PPL gap;
`max_prob` is the cleanest signal (matches AUROC 0.916).

This module is the *decision + orchestration* half (pure, env-gated, no graph
surgery). The *re-forward* itself is one extra CUDA-graph replay driven by the
model runner, which reads the updated `slot_table` and recomputes the promoted
experts at FP4. Everything is OFF unless `VLLM_MOE_W2_GATE=1`, so the prod
serving path is byte-for-byte unchanged by default.

Why the orchestration is out-of-graph (see CONFIDENCE_GATE_NEXT_SESSION.md):
the trigger (`max_prob` of THIS step's logits) is a runtime branch on a GPU
value, the forced promotion is synchronous + variable-size, and the re-run is a
2nd forward — none of which fit the captured one-graph-per-step cadence. The
re-forward CAN be a graph replay; only steps (a) read confidence, (b) force
promote, (c) trigger the replay are eager.

Env knobs:
  VLLM_MOE_W2_GATE         0 (default) | 1     master switch
  VLLM_MOE_W2_GATE_SIGNAL  max_prob (default) | margin
  VLLM_MOE_W2_GATE_TAU     fire if signal <= TAU. Default 0.60 for max_prob, 1.5
                           nats for margin. Pure quality<->latency knob. At 0.60
                           (measured, coding): fires ~16% of steps, precision ~46%
                           (FP4 differs from 2-bit there), 4.2x lift over the 10.8%
                           base disagreement, ~68% recall -- the efficiency knee
                           before added re-runs go mostly redundant. Raise toward
                           0.70-0.80 for more recall once a functional eval confirms
                           the FP4 upgrades are correct; lower to 0.50 if marginal.
  VLLM_MOE_W2_GATE_MAX_PROMOTE  cap experts force-promoted per fired step
                                (0 = unlimited, default).
  VLLM_MOE_W2_GATE_TRACE   0 (default) | 1 log each fire/re-forward.
"""

import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_ENABLED = os.getenv("VLLM_MOE_W2_GATE", "0") == "1"
_SIGNAL = os.getenv("VLLM_MOE_W2_GATE_SIGNAL", "max_prob")
_DEFAULT_TAU = {"max_prob": 0.60, "margin": 1.5}
_TAU = float(os.getenv("VLLM_MOE_W2_GATE_TAU", str(_DEFAULT_TAU.get(_SIGNAL, 0.60))))
_MAX_PROMOTE = int(os.getenv("VLLM_MOE_W2_GATE_MAX_PROMOTE", "0"))
_TRACE = os.getenv("VLLM_MOE_W2_GATE_TRACE", "0") == "1"
# Measurement mode: on a fired step, COUNT routed experts (delta._need) instead of
# promoting/re-forwarding -> study whether 2-bit difficulty concentrates on few
# experts. Zero serving perturbation; read [need] lines from the delta trace.
_CAPTURE = os.getenv("VLLM_MOE_W2_GATE_CAPTURE", "0") == "1"
# Optional runtime-tunable threshold: if VLLM_MOE_W2_GATE_TAU_FILE points at a
# file, its float contents override TAU (mtime-cached, re-read on change). Lets a
# threshold/latency sweep run in ONE server without restarts. A value that can
# never fire (e.g. max_prob<=0.0) effectively disables the gate (baseline).
_TAU_FILE = os.getenv("VLLM_MOE_W2_GATE_TAU_FILE", "")
_tau_dyn = _TAU
_tau_mtime = -1.0
# Diagnostic: when 0, a fired step force-promotes (warms cache) but SKIPS the
# 2nd forward — isolates re-forward correctness from force_promote. Default 1.
_REFORWARD = os.getenv("VLLM_MOE_W2_GATE_REFORWARD", "1") == "1"

# observability (cheap; only mutated when the gate is enabled)
_n_steps = 0
_n_fired = 0
_n_reforwarded = 0
_n_promoted = 0


def enabled() -> bool:
    return _ENABLED


def signal() -> str:
    return _SIGNAL


def _current_tau() -> float:
    """TAU, optionally overridden live by VLLM_MOE_W2_GATE_TAU_FILE (mtime-cached)."""
    global _tau_dyn, _tau_mtime
    if not _TAU_FILE:
        return _TAU
    try:
        m = os.path.getmtime(_TAU_FILE)
        if m != _tau_mtime:
            _tau_mtime = m
            with open(_TAU_FILE) as f:
                _tau_dyn = float(f.read().strip())
    except (OSError, ValueError):
        pass
    return _tau_dyn


def threshold() -> float:
    return _current_tau()


def reforward_enabled() -> bool:
    return _REFORWARD


def should_reforward(logits: torch.Tensor) -> bool:
    """Decide whether to re-forward this decode step at FP4.

    `logits` is the per-request next-token logits [num_reqs, vocab] from the
    1st (2-bit) forward. Fires when ANY request's top-1 is low-confidence -- the
    whole batch shares one CUDA graph, so a re-forward recomputes all rows
    together. Costs ONE GPU->CPU sync (the `.item()` below), incurred only when
    the gate is enabled.

    `margin` and `max_prob` are computed directly from logits without a full
    softmax: margin = top1_logit - top2_logit == log p1 - log p2 (the softmax
    normaliser cancels), and max_prob = exp(top1_logit - logsumexp(logits)).
    """
    global _n_steps, _n_fired
    _n_steps += 1
    if logits is None or logits.numel() == 0:
        return False
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    tau = _current_tau()
    top2 = torch.topk(logits, 2, dim=-1).values  # [R, 2]
    if _SIGNAL == "margin":
        worst = (top2[:, 0] - top2[:, 1]).min()
    else:  # max_prob
        lse = torch.logsumexp(logits, dim=-1)
        worst = torch.exp(top2[:, 0] - lse).min()
    fire = bool((worst <= tau).item())
    if fire:
        _n_fired += 1
        if _TRACE:
            logger.info("[gate] fire: %s worst=%.3f <= tau=%.3f (step %d)",
                        _SIGNAL, float(worst), tau, _n_steps)
    return fire


def force_promote_step(layers=None) -> int:
    """Pull this step's COLD routed experts up to FP4 via the delta tier.
    Returns the number promoted (0 if the tier is absent / nothing cold).

    MEASUREMENT mode (VLLM_MOE_W2_GATE_CAPTURE=1): instead of promoting, only
    COUNT this low-confidence step's routed experts (tier.mark_need_only) and
    return 0 -- so the caller skips the re-forward. Lets us study whether 2-bit
    difficulty concentrates on a small expert set with zero serving perturbation."""
    global _n_reforwarded, _n_promoted
    from vllm.model_executor.layers.quantization.utils import moe_w2_delta
    tier = moe_w2_delta._TIER
    if tier is None:
        return 0
    if _CAPTURE:
        tier.mark_need_only(layers=layers)
        return 0
    cap = _MAX_PROMOTE if _MAX_PROMOTE > 0 else None
    n = tier.force_promote(layers=layers, max_promote=cap)
    if n > 0:
        _n_reforwarded += 1
        _n_promoted += n
        if _TRACE:
            logger.info("[gate] force-promoted %d experts -> re-forward", n)
    return n


def stats() -> dict:
    return dict(steps=_n_steps, fired=_n_fired, reforwarded=_n_reforwarded,
                promoted=_n_promoted, signal=_SIGNAL, tau=_TAU,
                fire_rate=(_n_fired / _n_steps if _n_steps else 0.0))
