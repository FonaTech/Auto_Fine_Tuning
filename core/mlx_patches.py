"""
core/mlx_patches.py

Runtime monkey-patches to `mlx_tune` that reduce ORPO/DPO peak memory without
modifying the installed package. Safe no-ops when mlx_tune is unavailable (e.g.
CUDA machines) or when already patched.

Patches:
  1. `mlx_tune.losses.compute_log_probs`  /  `compute_log_probs_with_lengths`
     Use `logits[target] - logsumexp(logits)` instead of
     `take_along_axis(log_softmax(logits))`. Mathematically identical but
     avoids materialising a full `[B, S, V]` log_softmax tensor, which on
     Qwen3-4B (V≈152k, S=2048) saves ~2.5 GB per forward and much more in
     the retained backward graph.

  2. `mlx_tune.rl_trainers.{ORPOTrainer,DPOTrainer}._train_native` —
     wrap so `mx.clear_cache()` is invoked after each step's `mx.eval` to
     release the Metal allocator buffer and prevent fragmentation growth.
"""

from __future__ import annotations

from typing import Callable, Optional

LogFn = Callable[[str], None]

_APPLIED = False


def apply_mlx_tune_patches(log_fn: Optional[LogFn] = None) -> bool:
    """Apply the in-process patches. Returns True on success, False if skipped.

    Idempotent: re-calling is a no-op after the first successful application.
    """
    global _APPLIED
    if _APPLIED:
        return True

    def _log(msg: str) -> None:
        if log_fn:
            try:
                log_fn(msg)
            except Exception:
                pass

    try:
        import mlx.core as mx  # noqa: F401
    except ImportError:
        return False  # No MLX on this machine — nothing to patch

    applied_any = False

    # ── Patch 1: streaming log-prob in losses ────────────────────────
    try:
        from mlx_tune import losses as _losses

        def _log_probs_common(model, inputs, targets):
            logits = model(inputs)
            # gather logit at target tokens → [B, S]
            target_logits = mx.take_along_axis(
                logits, targets[:, :, None], axis=-1
            ).squeeze(-1)
            # numerically stable log(sum(exp(logits))) along vocab axis → [B, S]
            lse = mx.logsumexp(logits, axis=-1)
            return target_logits - lse  # log_softmax(logits)[target]

        def _patched_compute_log_probs(model, input_ids, attention_mask=None):
            inputs = input_ids[:, :-1]
            targets = input_ids[:, 1:]
            per_token = _log_probs_common(model, inputs, targets)
            if attention_mask is not None:
                mask = attention_mask[:, 1:]
                per_token = per_token * mask
            return per_token.sum(axis=-1)

        def _patched_compute_log_probs_with_lengths(model, input_ids, lengths):
            inputs = input_ids[:, :-1]
            targets = input_ids[:, 1:]
            per_token = _log_probs_common(model, inputs, targets)
            seq_len = targets.shape[1]
            positions = mx.arange(seq_len)[None, :]
            mask = positions < lengths[:, None]
            per_token = per_token * mask.astype(per_token.dtype)
            return per_token.sum(axis=-1)

        if hasattr(_losses, "compute_log_probs"):
            _losses.compute_log_probs = _patched_compute_log_probs
            applied_any = True
        if hasattr(_losses, "compute_log_probs_with_lengths"):
            _losses.compute_log_probs_with_lengths = (
                _patched_compute_log_probs_with_lengths
            )
            applied_any = True

        # Downstream modules that imported the symbols at module-load time
        # hold their own references — rebind those too.
        try:
            from mlx_tune import rl_trainers as _rl
            if hasattr(_rl, "compute_log_probs"):
                _rl.compute_log_probs = _patched_compute_log_probs
            if hasattr(_rl, "compute_log_probs_with_lengths"):
                _rl.compute_log_probs_with_lengths = (
                    _patched_compute_log_probs_with_lengths
                )
        except Exception:
            pass

        if applied_any:
            _log("[mlx_patches] compute_log_probs* patched to logsumexp path")
    except Exception as e:
        _log(f"[mlx_patches] log-probs patch skipped: {e}")

    # ── Patch 2: per-step clear_cache on preference trainers ─────────
    try:
        from mlx_tune import rl_trainers as _rl
        import mlx.core as mx

        def _wrap_train_native(cls, label: str):
            if getattr(cls, "_unsloth_gui_cache_patched", False):
                return
            original = cls._train_native

            def _wrapped(self, *args, **kwargs):
                # Intercept optimizer.update calls by patching the instance's
                # mx.eval to also clear the Metal allocator cache afterwards.
                real_eval = mx.eval

                def eval_then_clear(*eargs, **ekwargs):
                    out = real_eval(*eargs, **ekwargs)
                    try:
                        mx.clear_cache()
                    except Exception:
                        pass
                    return out

                _rl.mx.eval = eval_then_clear  # type: ignore[attr-defined]
                try:
                    return original(self, *args, **kwargs)
                finally:
                    _rl.mx.eval = real_eval  # type: ignore[attr-defined]

            cls._train_native = _wrapped
            cls._unsloth_gui_cache_patched = True  # type: ignore[attr-defined]
            _log(f"[mlx_patches] {label}._train_native wrapped with per-step clear_cache")

        for name in ("ORPOTrainer", "DPOTrainer"):
            cls = getattr(_rl, name, None)
            if cls is not None and hasattr(cls, "_train_native"):
                _wrap_train_native(cls, name)
                applied_any = True
    except Exception as e:
        _log(f"[mlx_patches] trainer cache patch skipped: {e}")

    if applied_any:
        _APPLIED = True
    return applied_any
