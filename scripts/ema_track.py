"""Trajectory EMA for the main (a0) run.

Keeps an fp32 exponential moving average of the trainable weights along the a0 trajectory,
updated every ``every`` optimizer steps with a time constant equal to ``horizon`` of the
planned step count. At ``on_train_end`` (installed ahead of the soup callback) the EMA
iterate is written into the model, scored on the dev split, offered to the soup pool as an
extra candidate, and adopted as the best single state when it beats the best raw snapshot.
The raw weights are restored afterwards so the soup/dev-pass flow continues unchanged.

Why: the a0 horizon is planned with an overshoot and a warmup-stable-decay floor, so the
run is usually stopped by the clock before the decay finishes. The EMA iterate is a cheap,
always-ready annealed point along the trajectory that only costs one dev evaluation; the
dev split decides whether it, the raw best, or a soup containing it ships.
"""
import math
import os
import time
from typing import Optional

import torch
from transformers import TrainerCallback

import host_snapshot

ENV_ON = "SN56_EMA"
ENV_EVERY = "SN56_EMA_EVERY"
ENV_HORIZON = "SN56_EMA_HORIZON"
ENV_DEVICE = "SN56_EMA_DEVICE"
_FALSY = ("0", "false", "no", "off")

DEFAULT_EVERY = 4
DEFAULT_HORIZON = 0.15
MIN_UPDATES = 8
GPU_MARGIN_BYTES = 6 << 30
HOST_MARGIN_BYTES = 8 << 30
MAX_UPDATE_SHARE = 0.03
MAX_EVERY = 64
MIN_SECS_LEFT = 300


def enabled() -> bool:
    return os.environ.get(ENV_ON, "1").strip().lower() not in _FALSY


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


def _unwrap(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def _host_available_bytes() -> Optional[int]:
    try:
        return host_snapshot.available_ram_bytes()
    except Exception:
        return None


def _seconds_left(end_time) -> Optional[float]:
    """Seconds until the trainer's UTC deadline string ('%Y-%m-%d %H:%M:%S'), None if unknown."""
    if not end_time:
        return None
    try:
        from datetime import datetime, timezone
        end = datetime.strptime(str(end_time), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        return (end - datetime.now(timezone.utc)).total_seconds()
    except Exception:
        return None


def decay_for(planned_steps: int, every: int, horizon: float, min_updates: int = MIN_UPDATES) -> float:
    """Per-update decay so that the EMA time constant spans ``horizon`` of the planned run
    (never shorter than ``min_updates`` updates)."""
    tau = max(float(min_updates * every), float(horizon) * float(max(1, planned_steps)))
    return math.exp(-float(every) / tau)


class EMATracker(TrainerCallback):
    def __init__(self, ckpt_avg, *, every: Optional[int] = None, horizon: Optional[float] = None,
                 device: Optional[str] = None, log=print):
        self.ckpt_avg = ckpt_avg
        self.every = max(1, int(every or _env_int(ENV_EVERY, DEFAULT_EVERY)))
        self.horizon = min(0.9, max(0.02, float(horizon or _env_float(ENV_HORIZON, DEFAULT_HORIZON))))
        self.device_pref = (device or os.environ.get(ENV_DEVICE, "auto")).strip().lower()
        self.log = log
        self.names: list = []
        self.ema: Optional[list] = None
        self.where: str = "off"
        self.n_updates = 0
        self.secs_update = 0.0
        self.disabled_reason: Optional[str] = None
        self.offered: Optional[dict] = None
        self._t_last_step: Optional[float] = None
        self._step_secs: Optional[float] = None
        self._armed = False
        self._initialised = False

    # ----- placement -----
    @staticmethod
    def _trainable(model):
        return [(n, p) for n, p in _unwrap(model).named_parameters() if p.requires_grad]

    def _place(self, named):
        need = sum(int(p.numel()) for _, p in named) * 4
        dev0 = named[0][1].device
        if self.device_pref in ("auto", "cuda") and dev0.type == "cuda":
            try:
                free, _total = torch.cuda.mem_get_info(dev0)
                if free - need >= GPU_MARGIN_BYTES:
                    return dev0, need
            except Exception:
                pass
            if self.device_pref == "cuda":
                return None, need
        avail = _host_available_bytes()
        if avail is None:
            return torch.device("cpu"), need
        host_margin = min(HOST_MARGIN_BYTES, max(1 << 30, int(0.2 * avail)))
        if avail - need >= host_margin:
            return torch.device("cpu"), need
        return None, need

    # ----- callbacks -----
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self._armed = model is not None
        self._initialised = False
        if not self._armed:
            self.disabled_reason = "no_model"

    def _initialise(self, model) -> bool:
        """Allocate the EMA buffers at the first qualifying step, when optimizer states and the
        activation peak already exist so the free-memory reading is honest."""
        self._initialised = True
        try:
            named = self._trainable(model)
            if not named:
                self.disabled_reason = "no_trainable"
                return False
            if any(hasattr(p, "ds_id") or hasattr(p, "_is_flat_param") for _, p in named):
                self.disabled_reason = "sharded"
                return False
            dev, need = self._place(named)
            if dev is None:
                self.disabled_reason = f"no_room({need / 1e9:.1f}GB)"
                self.log(f"[justice][ema] disabled: {self.disabled_reason}")
                return False
            try:
                with torch.no_grad():
                    self.ema = [p.detach().to(device=dev, dtype=torch.float32, copy=True) for _, p in named]
            except torch.OutOfMemoryError:
                self.ema = None
                torch.cuda.empty_cache()
                if dev.type != "cuda":
                    raise
                dev = torch.device("cpu")
                self.log("[justice][ema] GPU OOM while allocating; falling back to host RAM")
                with torch.no_grad():
                    self.ema = [p.detach().to(device=dev, dtype=torch.float32, copy=True) for _, p in named]
            self.names = [n for n, _ in named]
            self.where = str(dev)
            self.n_updates = 0
            self.log(f"[justice][ema] tracking {len(named)} tensors ({need / 1e9:.2f}GB fp32) on "
                     f"{self.where}; every={self.every} steps, horizon={self.horizon:.2f} of plan")
            return True
        except Exception as exc:
            self.ema = None
            self.disabled_reason = f"init_error:{type(exc).__name__}"
            self.log(f"[justice][ema] disabled: {self.disabled_reason}: {exc}")
            return False

    def _decay(self, state) -> float:
        planned = int(getattr(state, "max_steps", 0) or 0)
        if planned <= 0:
            planned = max(1, int(state.global_step)) * 4
        return decay_for(planned, self.every, self.horizon)

    @torch.no_grad()
    def _update(self, model, state) -> None:
        d = self._decay(state)
        one_minus = 1.0 - d
        params = {n: p for n, p in _unwrap(model).named_parameters() if p.requires_grad}
        on_cuda = self.where.startswith("cuda")
        for n, e in zip(self.names, self.ema):
            p = params.get(n)
            if p is None:
                continue
            src = p.data if (on_cuda or p.device.type == "cpu") else p.data.detach().to("cpu", dtype=torch.float32)
            e.mul_(d).add_(src, alpha=one_minus)
        self.n_updates += 1

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None or not getattr(self, "_armed", False):
            return
        if not getattr(self, "_initialised", False):
            if int(state.global_step) % self.every != 0:
                return
            if not self._initialise(model):
                self._armed = False
                return
        if self.ema is None:
            return
        now = time.perf_counter()
        if self._t_last_step is not None:
            dt = now - self._t_last_step
            self._step_secs = dt if self._step_secs is None else 0.9 * self._step_secs + 0.1 * dt
        self._t_last_step = now
        if int(state.global_step) % self.every != 0:
            return
        try:
            t0 = time.perf_counter()
            self._update(model, state)
            secs = time.perf_counter() - t0
            self.secs_update = secs
            if self.where == "cpu" and self._step_secs:
                if secs > MAX_UPDATE_SHARE * self._step_secs * self.every:
                    new_every = min(MAX_EVERY, int(math.ceil(secs / max(1e-6, MAX_UPDATE_SHARE * self._step_secs))))
                    if new_every > self.every:
                        self.log(f"[justice][ema] host update {secs:.2f}s; stretching cadence "
                                 f"{self.every} -> {new_every} steps")
                        self.every = new_every
        except Exception as exc:
            self.disabled_reason = f"update_error:{type(exc).__name__}"
            self.log(f"[justice][ema] disabled after {self.n_updates} updates: {exc}")
            self.ema = None

    @torch.no_grad()
    def _load_into(self, model) -> None:
        params = {n: p for n, p in _unwrap(model).named_parameters() if p.requires_grad}
        for n, e in zip(self.names, self.ema):
            p = params.get(n)
            if p is None:
                continue
            p.data.copy_(e.to(device=p.device, dtype=p.dtype))

    def _offer(self, model, state) -> None:
        ca = self.ckpt_avg
        if self.ema is None or model is None:
            self.log(f"[justice][ema] nothing to offer ({self.disabled_reason})")
            return
        if self.n_updates < MIN_UPDATES:
            self.log(f"[justice][ema] only {self.n_updates} updates; not offered")
            return
        if ca is None or getattr(ca, "trainer", None) is None or getattr(ca, "averaging_mode", "") != "ram":
            self.log("[justice][ema] soup callback unavailable; not offered")
            return
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.log("[justice][ema] multi-process run; not offered")
            return
        left = _seconds_left(getattr(ca, "end_time", None))
        if left is not None and left < MIN_SECS_LEFT:
            self.log(f"[justice][ema] only {left:.0f}s left before the deadline; not offered")
            return
        step = int(state.global_step)
        raw = ca._snapshot(model, step=step, consumer="ema.raw")
        host_snapshot.weights_changed("ema.load")
        t0 = time.perf_counter()
        try:
            self._load_into(model)
            loss = ca._eval_synced(model)
            raw_best = float(ca.best_loss) if ca.best_loss is not None else float("inf")
            ok = loss is not None and loss == loss and loss != float("inf")
            self.offered = {"loss": (float(loss) if ok else None), "raw_best": raw_best,
                            "n_updates": int(self.n_updates), "step": step, "where": self.where,
                            "every": int(self.every), "horizon": float(self.horizon)}
            if not ok:
                self.log("[justice][ema] dev eval failed; not offered")
                return
            prev_tag = ca.pool_tag
            ca.pool_tag = "ema"
            try:
                ca._consider_for_pool(model, float(loss), step)
            finally:
                ca.pool_tag = prev_tag
            adopted = float(loss) < raw_best
            if adopted:
                ca.best_state = ca._snapshot(model, step=step, consumer="ema.best")
                ca.best_loss = float(loss)
                ca.best_step = step
                ca.best_source = f"ema(n={self.n_updates},every={self.every},h={self.horizon:.2f})"
            self.log(f"[justice][ema] dev {float(loss):.4f} vs best raw {raw_best:.4f} -> "
                     f"{'adopted as best' if adopted else 'pooled only'} "
                     f"({self.n_updates} updates, {time.perf_counter() - t0:.0f}s)")
        finally:
            ca._restore(model, raw)
            raw = None

    def on_train_end(self, args, state, control, model=None, **kwargs):
        try:
            self._offer(model, state)
        except Exception as exc:
            self.log(f"[justice][ema] offer failed ({type(exc).__name__}: {exc})")
        finally:
            self.ema = None
            if self.where.startswith("cuda"):
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass


def install(trainer, tracker: EMATracker, before=None) -> EMATracker:
    """Insert the tracker so its ``on_train_end`` runs before ``before`` (the soup callback)."""
    cbs = trainer.callback_handler.callbacks
    idx = None
    if before is not None:
        for i, cb in enumerate(cbs):
            if cb is before:
                idx = i
                break
    if idx is None:
        cbs.append(tracker)
    else:
        cbs.insert(idx, tracker)
    return tracker
