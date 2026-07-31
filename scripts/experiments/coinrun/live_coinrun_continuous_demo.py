#!/usr/bin/env python3
"""Continuous browser demo for action-conditioned CoinRun world-model inference."""

from __future__ import annotations

import argparse
import base64
import functools
import json
import operator
import os
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import hydra
import imageio.v3 as iio
import jax
import jax.numpy as jnp
from flax import nnx
from omegaconf import OmegaConf

from dreamer.actions import Actions, shift_actions
from dreamer.checkpointing import DynamicsCheckpointBundle
from dreamer.data import build_iterator
from dreamer.generation import DenoiseSchedule, next_frame
from dreamer.parallel import build_parallel
from dreamer.sampler import encode_jit
from dreamer.utils import normalize_latents


os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

for name, resolver in (
    ("mul", lambda *args: functools.reduce(operator.mul, args)),
    ("sum", lambda *args: sum(args)),
    ("floordiv", lambda x, y: x // y),
    ("max", lambda *args: max(args)),
):
    if not OmegaConf.has_resolver(name):
        OmegaConf.register_new_resolver(name, resolver)


ACTION_NAMES = {
    1: "左",
    2: "左 + 跳",
    4: "停 / No-op",
    5: "跳",
    7: "右",
    8: "右 + 跳",
}

VALID_INPUTS = frozenset({"left", "right", "jump"})


def resolve_pressed_action(pressed: set[str]) -> int:
    """Resolve a complete held-input state to the CoinRun categorical action."""
    unsupported = set(pressed) - VALID_INPUTS
    if unsupported:
        raise ValueError(f"Unsupported input: {sorted(unsupported)}")
    left = "left" in pressed
    right = "right" in pressed
    jump = "jump" in pressed
    if left and right:
        left = False
        right = False
    if left and jump:
        return 2
    if right and jump:
        return 8
    if left:
        return 1
    if right:
        return 7
    if jump:
        return 5
    return 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-name", default="best")
    parser.add_argument("--context", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--denoise-steps", type=int, default=4)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


def png_data_url(frame: jax.Array) -> str:
    frame_host = jax.device_get(jnp.clip(frame, 0, 255).astype(jnp.uint8))
    encoded = iio.imwrite("<bytes>", frame_host, extension=".png")
    return "data:image/png;base64," + base64.b64encode(encoded).decode("ascii")


def append_context_latent(
    latents_ctx: jax.Array, next_latent: jax.Array
) -> jax.Array:
    """Append one generated latent without changing the context buffer dtype."""
    next_latent = next_latent.astype(latents_ctx.dtype)
    return jnp.concatenate([latents_ctx[:, 1:], next_latent], axis=1)


def resolve_config_dir(script_path: Path) -> Path:
    return script_path.resolve().parents[3] / "configs"


@nnx.jit
def prefill_dynamics_jit(
    dynamics,
    actions_ctx,
    normalized_latents,
    step_indices,
    tau_indices,
    caches,
):
    _, (_, updated_caches) = dynamics(
        actions_ctx,
        step_indices,
        tau_indices,
        normalized_latents,
        task_embeddings=None,
        caches=caches,
        deterministic=True,
    )
    return updated_caches


@nnx.jit
def prefill_decoder_jit(tokenizer, latents_ctx, caches):
    frames, updated_caches = tokenizer.decode(
        latents_ctx, caches=caches, deterministic=True
    )
    return frames, updated_caches


def compile_next_frame(schedule: DenoiseSchedule, latent_shape: tuple[int, ...]):
    """Compile one persistent-cache frame step with static sampling metadata."""

    @nnx.jit
    def next_frame_jit(
        tokenizer,
        dynamics,
        action,
        dynamics_cache,
        decoder_cache,
        rng,
    ):
        return next_frame(
            tokenizer=tokenizer,
            dynamics=dynamics,
            schedule=schedule,
            action=action,
            latent_shape=latent_shape,
            dynamics_cache=dynamics_cache,
            decoder_cache=decoder_cache,
            rng=rng,
        )

    return next_frame_jit


class CoinRunWorld:
    def __init__(
        self,
        *,
        checkpoint: Path,
        dataset: Path,
        model_name: str,
        context: int,
        seed: int,
        denoise_steps: int,
    ) -> None:
        self.model_name = model_name
        self.context = context
        self.seed = seed
        self.condition = threading.Condition()
        self.model_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.subscribers = 0
        self.paused = False
        self.reset_requested = False
        self.input_action = 4
        self.input_revision = -1
        self.state_version = 0
        self.status = "loading"
        self.error: str | None = None
        self.worker: threading.Thread | None = None

        config_dir = resolve_config_dir(Path(__file__))
        with hydra.initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            self.cfg = hydra.compose(
                config_name="eval_fvd",
                overrides=[
                    "dataset=coinrun",
                    f"dataset.array_record_path={dataset}",
                    "dataset.dataloader_cfg.B=1",
                    f"dataset.dataloader_cfg.long_T={context}",
                    "dataset.dataloader_cfg.num_workers=0",
                    f"seed={seed}",
                    "parallel_strategy=data",
                ],
            )

        self.mesh, self.data_sharding, mesh_rules = build_parallel(
            self.cfg.parallel_strategy
        )
        self.mesh_context = jax.set_mesh(self.mesh)
        self.mesh_context.__enter__()

        checkpoint = checkpoint.resolve()
        if checkpoint.name != "checkpoints" and (checkpoint / "checkpoints").is_dir():
            checkpoint = checkpoint / "checkpoints"
        self.checkpoint = checkpoint
        self.bundle = DynamicsCheckpointBundle.from_pretrained(
            str(checkpoint),
            mesh_rules=mesh_rules,
            model_names={"dynamics_ema", "tokenizer"},
        )
        self.tokenizer = self.bundle.tokenizer
        self.dynamics = self.bundle.dynamics_ema
        self.schedule = DenoiseSchedule.init(
            denoise_steps, self.dynamics.cfg.k_max
        )
        self.dataloader = iter(
            build_iterator(
                self.cfg.dataset,
                seed=seed,
                device=self.data_sharding,
                return_actions=True,
            )
        )
        self.rng = jax.random.PRNGKey(seed)
        self.next_frame_jit = None
        self.step_count = 0
        self.last_action = 4
        self.last_latency_ms = 0.0
        self.current_frame_url = ""
        self._load_next_context()
        self.warmup_ms = self._warmup()
        self.status = "idle"
        self._publish()

    def _warmup(self) -> float:
        saved_frames = self.context_frames
        saved_actions = self.context_actions
        saved_rng = self.rng
        started = time.perf_counter()
        self._step_once(4)
        jax.block_until_ready(self.current_frame)
        elapsed = (time.perf_counter() - started) * 1000.0
        self.rng = saved_rng
        self._initialize_context(saved_frames, saved_actions)
        self.last_latency_ms = 0.0
        return elapsed

    def _load_next_context(self) -> None:
        batch = next(self.dataloader)
        self._initialize_context(batch["videos"], batch["actions"])

    def _initialize_context(self, frames, actions) -> None:
        self.context_frames = frames
        self.context_actions = actions
        self.latents_ctx = encode_jit(self.tokenizer, frames)
        self.actions_ctx = shift_actions(
            actions,
            self.cfg.dataset.categorical_action_dim,
            self.cfg.dataset.categorical_noop_action,
        )
        normalized = normalize_latents(
            self.latents_ctx,
            self.dynamics.cfg.latent_mean,
            self.dynamics.cfg.latent_std,
        )
        batch_size, time_steps, n_latents, d_bottleneck = normalized.shape
        dynamics_window = int(
            self.dynamics.cfg.context_length or max(time_steps, self.context)
        )
        dynamics_cache = self.dynamics.create_static_caches(
            batch_size=batch_size,
            n_latents=n_latents,
            window_size=dynamics_window,
            n_agent=0,
            dtype=normalized.dtype,
        )
        step_indices = jnp.full(
            (batch_size, time_steps), self.schedule.emax, dtype=jnp.int32
        )
        tau_indices = jnp.full(
            (batch_size, time_steps), self.schedule.k_max, dtype=jnp.int32
        )
        self.dynamics_cache = prefill_dynamics_jit(
            self.dynamics,
            self.actions_ctx,
            normalized,
            step_indices,
            tau_indices,
            dynamics_cache,
        )
        decoder_window = int(
            self.tokenizer.decoder.context_length or max(time_steps, self.context)
        )
        decoder_cache = self.tokenizer.decoder.create_static_caches(
            batch_size=batch_size,
            window_size=decoder_window,
            dtype=self.latents_ctx.dtype,
        )
        _, self.decoder_cache = prefill_decoder_jit(
            self.tokenizer, self.latents_ctx, decoder_cache
        )
        latent_shape = (batch_size, 1, n_latents, d_bottleneck)
        if self.next_frame_jit is None:
            self.next_frame_jit = compile_next_frame(self.schedule, latent_shape)
        self.current_frame = jnp.clip(frames[0, -1], 0, 255).astype(jnp.uint8)
        self.current_frame_url = png_data_url(self.current_frame)
        self.step_count = 0
        self.last_action = 4
        self.last_latency_ms = 0.0
        self.error = None

    def _step_once(self, action_id: int) -> None:
        action = Actions(categorical=jnp.asarray([action_id], dtype=jnp.int32))
        next_frame_jit = self.next_frame_jit
        if next_frame_jit is None:
            raise RuntimeError("Interactive frame step was not initialized")
        (
            frame,
            _,
            self.dynamics_cache,
            self.decoder_cache,
            self.rng,
        ) = next_frame_jit(
            self.tokenizer,
            self.dynamics,
            action,
            self.dynamics_cache,
            self.decoder_cache,
            self.rng,
        )
        self.current_frame = frame[0, -1]
        self.step_count += 1
        self.last_action = action_id

    def _snapshot_unlocked(self) -> dict[str, Any]:
        is_streaming = (
            self.subscribers > 0
            and not self.paused
            and self.error is None
            and self.status != "loading"
        )
        return {
            "frame": self.current_frame_url,
            "model": self.model_name,
            "protocol": "continuous-sse-v1",
            "context_frames": self.context,
            "denoise_steps": self.schedule.num_steps,
            "step": self.step_count,
            "action_id": self.last_action,
            "action": ACTION_NAMES[self.last_action],
            "input_action_id": self.input_action,
            "input_action": ACTION_NAMES[self.input_action],
            "latency_ms": round(self.last_latency_ms, 1),
            "warmup_ms": round(getattr(self, "warmup_ms", 0.0), 1),
            "fps": round(1000.0 / self.last_latency_ms, 2)
            if self.last_latency_ms > 0
            else 0.0,
            "paused": self.paused,
            "streaming": is_streaming,
            "subscribers": self.subscribers,
            "status": self.status,
            "error": self.error,
            "version": self.state_version,
        }

    def snapshot(self) -> dict[str, Any]:
        with self.condition:
            return self._snapshot_unlocked()

    def _publish(self) -> None:
        with self.condition:
            self.state_version += 1
            self.condition.notify_all()

    def set_input(
        self, pressed: set[str], revision: int
    ) -> dict[str, Any]:
        action_id = resolve_pressed_action(pressed)
        with self.condition:
            if revision > self.input_revision:
                self.input_revision = revision
                self.input_action = action_id
                self.state_version += 1
                self.condition.notify_all()
            return self._snapshot_unlocked()

    def set_paused(self, paused: bool) -> dict[str, Any]:
        with self.condition:
            self.paused = paused
            if paused:
                self.input_action = 4
            self.state_version += 1
            self.condition.notify_all()
            return self._snapshot_unlocked()

    def request_reset(self) -> dict[str, Any]:
        with self.condition:
            self.reset_requested = True
            self.status = "resetting"
            self.input_action = 4
            self.state_version += 1
            self.condition.notify_all()
            return self._snapshot_unlocked()

    def register_subscriber(self) -> None:
        with self.condition:
            self.subscribers += 1
            self.state_version += 1
            self.condition.notify_all()

    def unregister_subscriber(self) -> None:
        with self.condition:
            self.subscribers = max(0, self.subscribers - 1)
            self.input_action = 4
            self.state_version += 1
            self.condition.notify_all()

    def wait_for_update(
        self, after_version: int, timeout: float
    ) -> dict[str, Any] | None:
        with self.condition:
            changed = self.condition.wait_for(
                lambda: self.state_version > after_version
                or self.stop_event.is_set(),
                timeout=timeout,
            )
            if not changed or self.stop_event.is_set():
                return None
            return self._snapshot_unlocked()

    def _run(self) -> None:
        while not self.stop_event.is_set():
            with self.condition:
                self.condition.wait_for(
                    lambda: self.stop_event.is_set()
                    or self.reset_requested
                    or (self.subscribers > 0 and not self.paused)
                )
                if self.stop_event.is_set():
                    return
                should_reset = self.reset_requested
                if should_reset:
                    self.reset_requested = False
                action_id = self.input_action

            try:
                with self.model_lock:
                    if should_reset:
                        self._load_next_context()
                        self.status = "paused" if self.paused else "running"
                    else:
                        started = time.perf_counter()
                        self._step_once(action_id)
                        jax.block_until_ready(self.current_frame)
                        self.last_latency_ms = (
                            time.perf_counter() - started
                        ) * 1000.0
                        self.current_frame_url = png_data_url(self.current_frame)
                        self.status = "running"
                self._publish()
            except Exception as exc:
                with self.condition:
                    self.error = f"{type(exc).__name__}: {exc}"
                    self.status = "error"
                    self.paused = True
                    self.state_version += 1
                    self.condition.notify_all()

    def start(self) -> None:
        if self.worker is not None:
            return
        self.worker = threading.Thread(
            target=self._run,
            name="coinrun-continuous-world",
            daemon=True,
        )
        self.worker.start()

    def stop(self) -> None:
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()
        if self.worker is not None:
            self.worker.join(timeout=5.0)


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>OpenDreamer CoinRun Live Demo</title>
  <style>
    :root { color-scheme:dark; --bg:#080a0d; --panel:#11151b; --panel-2:#171c24;
      --line:#2b3440; --ink:#f3f6f8; --muted:#929daa; --green:#76e6a6;
      --green-dim:#183b2a; --orange:#f6b86b; --danger:#ff8c8c; }
    * { box-sizing:border-box; }
    html,body { margin:0; width:100%; height:100%; background:var(--bg); }
    body { height:100dvh; overflow:hidden; padding:14px; color:var(--ink);
      font:14px/1.4 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
    button { font:inherit; }
    main { width:min(1120px,100%); height:100%; margin:0 auto; display:grid;
      grid-template-rows:52px minmax(0,1fr); gap:10px; }
    .topbar { display:flex; align-items:center; justify-content:space-between; gap:16px;
      border-bottom:1px solid var(--line); min-width:0; }
    .brand { min-width:0; display:flex; align-items:baseline; gap:10px; }
    h1 { margin:0; font-size:20px; line-height:1; letter-spacing:-.025em; white-space:nowrap; }
    .subtitle { color:var(--muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .live { flex:none; display:flex; align-items:center; gap:7px; color:var(--muted);
      font-size:12px; }
    .dot { width:8px; height:8px; border-radius:50%; background:var(--orange); }
    .live.connected .dot { background:var(--green); box-shadow:0 0 0 4px #76e6a61a; }
    .live.error { color:var(--danger); }
    .live.error .dot { background:var(--danger); }
    .workspace { min-height:0; display:grid; grid-template-columns:minmax(0,1fr) 240px; gap:12px; }
    .stage { min-width:0; min-height:0; display:grid; place-items:center; border:1px solid var(--line);
      border-radius:16px; background:#050709; overflow:hidden; position:relative; }
    .screen { width:min(100%,calc(100dvh - 92px)); max-width:720px;
      max-height:calc(100dvh - 92px); aspect-ratio:1/1; position:relative; background:#020304; }
    .screen img { display:block; width:100%; height:100%; object-fit:contain; image-rendering:pixelated; }
    .screen.loading img { opacity:.12; }
    .loader { position:absolute; inset:0; display:none; place-items:center; color:var(--muted);
      text-align:center; padding:24px; }
    .screen.loading .loader { display:grid; }
    .loader b { display:block; color:var(--ink); margin-bottom:4px; }
    .overlay { position:absolute; inset:auto 10px 10px 10px; display:flex;
      justify-content:space-between; align-items:flex-end; pointer-events:none; gap:8px; }
    .overlay-chip { border:1px solid #ffffff1f; border-radius:8px; background:#080a0dcc;
      color:#dce3e8; padding:5px 8px; font-size:12px; backdrop-filter:blur(8px); }
    aside { min-height:0; border:1px solid var(--line); border-radius:16px; background:var(--panel);
      padding:12px; display:flex; flex-direction:column; gap:12px; overflow:auto; }
    .section-title { color:var(--muted); font-size:11px; font-weight:700;
      letter-spacing:.1em; text-transform:uppercase; margin-bottom:7px; }
    .controls { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:7px; }
    .action { min-height:46px; border:1px solid var(--line); border-radius:10px;
      background:var(--panel-2); color:var(--ink); font-weight:650; cursor:pointer;
      touch-action:none; user-select:none; }
    .action:hover { border-color:#536273; }
    .action.active { background:var(--green-dim); border-color:var(--green); color:#caffdf; }
    .action:focus-visible,.utility:focus-visible { outline:2px solid var(--green); outline-offset:2px; }
    .stats { display:grid; gap:1px; border:1px solid var(--line); border-radius:10px; overflow:hidden; }
    .stat { min-height:34px; padding:7px 9px; display:flex; align-items:center;
      justify-content:space-between; gap:8px; background:#0c1015; }
    .stat + .stat { border-top:1px solid var(--line); }
    .stat span { color:var(--muted); }
    .stat b { font-weight:650; text-align:right; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .utilities { margin-top:auto; display:grid; grid-template-columns:1fr 1fr; gap:7px; }
    .utility { min-height:40px; border:1px solid var(--line); border-radius:10px;
      background:#0c1015; color:var(--ink); cursor:pointer; }
    .utility:hover { background:var(--panel-2); }
    .reset { color:var(--orange); }
    .hint { margin:0; color:var(--muted); font-size:12px; }
    .error-box { display:none; border:1px solid #ff8c8c55; border-radius:10px;
      background:#321b1b; color:#ffc2c2; padding:9px; font-size:12px; overflow-wrap:anywhere; }
    .error-box.visible { display:block; }
    @media(max-width:760px) {
      body { height:auto; min-height:100dvh; overflow:auto; padding:10px; }
      main { height:auto; grid-template-rows:44px auto; }
      .subtitle { display:none; }
      .workspace { grid-template-columns:1fr; }
      .stage { min-height:280px; }
      .screen { width:min(100%,calc(100dvh - 290px)); min-width:280px;
        max-height:calc(100dvh - 290px); }
      aside { overflow:visible; }
      .controls { grid-template-columns:repeat(3,minmax(0,1fr)); }
    }
    @media(prefers-reduced-motion:no-preference) {
      .dot { transition:background-color .18s ease,box-shadow .18s ease; }
      .action,.utility { transition:background-color .12s ease,border-color .12s ease; }
    }
  </style>
</head>
<body>
<main>
  <header class="topbar">
    <div class="brand">
      <h1>CoinRun World Model</h1>
      <span class="subtitle">持续生成，按键更新当前动作</span>
    </div>
    <div class="live" id="connection"><span class="dot"></span><span id="connectionText">正在连接</span></div>
  </header>
  <div class="workspace">
    <section class="stage" aria-label="模型生成画面">
      <div class="screen loading" id="screen">
        <img id="frame" alt="CoinRun 世界模型生成画面">
        <div class="loader"><div><b>正在加载模型画面</b><span>连接后世界会自动向前生成</span></div></div>
        <div class="overlay">
          <span class="overlay-chip" id="streamState">等待推流</span>
          <span class="overlay-chip"><span id="fps">0.00</span> FPS</span>
        </div>
      </div>
    </section>
    <aside>
      <section>
        <div class="section-title">持续输入</div>
        <div class="controls" id="controls">
          <button class="action" data-keys="left">← 左</button>
          <button class="action" data-keys="right">右 →</button>
          <button class="action" data-keys="jump">↑ 跳</button>
          <button class="action" data-keys="left,jump">↖ 左跳</button>
          <button class="action" data-keys="">· 松开</button>
          <button class="action" data-keys="right,jump">右跳 ↗</button>
        </div>
      </section>
      <section>
        <div class="section-title">运行状态</div>
        <div class="stats">
          <div class="stat"><span>模型</span><b id="model">等待</b></div>
          <div class="stat"><span>生成帧</span><b id="step">0</b></div>
          <div class="stat"><span>采用动作</span><b id="action">No-op</b></div>
          <div class="stat"><span>当前输入</span><b id="inputAction">No-op</b></div>
          <div class="stat"><span>单帧耗时</span><b id="latency">等待</b></div>
          <div class="stat"><span>历史 / 采样</span><b><span id="ctx">0</span> / <span id="denoise">0</span></b></div>
        </div>
      </section>
      <div class="error-box" id="errorBox" role="alert"></div>
      <p class="hint">键盘支持 ←、→、空格。按住期间，后端持续用同一输入生成；松开立即恢复 No-op。</p>
      <div class="utilities">
        <button class="utility" id="pause">暂停</button>
        <button class="utility reset" id="reset">换历史</button>
      </div>
    </aside>
  </div>
</main>
<script>
const frame = document.querySelector('#frame');
const screen = document.querySelector('#screen');
const controls = document.querySelector('#controls');
const keyboardKeys = new Set();
let pointerKeys = new Set();
let pointerActive = false;
let inputCounter = 0;
let paused = false;

function render(s) {
  if (s.frame) {
    frame.src = s.frame;
    screen.classList.remove('loading');
  }
  document.querySelector('#model').textContent = s.model;
  document.querySelector('#ctx').textContent = s.context_frames;
  document.querySelector('#denoise').textContent = s.denoise_steps;
  document.querySelector('#step').textContent = s.step;
  document.querySelector('#action').textContent = s.action;
  document.querySelector('#inputAction').textContent = s.input_action;
  document.querySelector('#latency').textContent = s.latency_ms ? `${s.latency_ms} ms` : '等待';
  document.querySelector('#fps').textContent = Number(s.fps || 0).toFixed(2);
  document.querySelector('#streamState').textContent =
    s.status === 'resetting' ? '正在重置' : s.paused ? '已暂停' : s.streaming ? '持续生成' : '等待画面';
  paused = Boolean(s.paused);
  document.querySelector('#pause').textContent = paused ? '继续' : '暂停';
  const errorBox = document.querySelector('#errorBox');
  errorBox.textContent = s.error || '';
  errorBox.classList.toggle('visible', Boolean(s.error));
}

function setConnection(state, text) {
  const node = document.querySelector('#connection');
  node.className = `live ${state}`;
  document.querySelector('#connectionText').textContent = text;
}

async function post(path, body) {
  try {
    const r = await fetch(path, {method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify(body), keepalive:true});
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || r.statusText);
    render(data);
    return data;
  } catch (error) {
    setConnection('error', '控制请求失败');
    const errorBox = document.querySelector('#errorBox');
    errorBox.textContent = error.message;
    errorBox.classList.add('visible');
    throw error;
  }
}

function pressedState() {
  return [...new Set([...keyboardKeys, ...pointerKeys])].sort();
}

function renderPressed() {
  const active = new Set(pressedState());
  for (const button of controls.querySelectorAll('.action')) {
    const keys = button.dataset.keys ? button.dataset.keys.split(',') : [];
    button.classList.toggle('active',
      keys.length === active.size && keys.every(key => active.has(key)));
  }
}

function sendInput() {
  renderPressed();
  const revision = Date.now() * 1000 + (++inputCounter % 1000);
  post('/api/input', {pressed:pressedState(), revision}).catch(() => {});
}

for (const button of controls.querySelectorAll('.action')) {
  button.addEventListener('pointerdown', event => {
    event.preventDefault();
    pointerActive = true;
    pointerKeys = new Set(button.dataset.keys ? button.dataset.keys.split(',') : []);
    button.setPointerCapture?.(event.pointerId);
    sendInput();
  });
}

function releasePointer() {
  if (!pointerActive) return;
  pointerActive = false;
  pointerKeys.clear();
  sendInput();
}
addEventListener('pointerup', releasePointer);
addEventListener('pointercancel', releasePointer);

const keyMap = {ArrowLeft:'left', ArrowRight:'right', Space:'jump'};
addEventListener('keydown', event => {
  const key = keyMap[event.code];
  if (!key) return;
  event.preventDefault();
  if (event.repeat || keyboardKeys.has(key)) return;
  keyboardKeys.add(key);
  sendInput();
});
addEventListener('keyup', event => {
  const key = keyMap[event.code];
  if (!key) return;
  event.preventDefault();
  keyboardKeys.delete(key);
  sendInput();
});
addEventListener('blur', () => {
  keyboardKeys.clear();
  pointerKeys.clear();
  sendInput();
});

document.querySelector('#pause').onclick = () =>
  post('/api/control', {paused:!paused}).catch(() => {});
document.querySelector('#reset').onclick = () =>
  post('/api/reset', {}).catch(() => {});

const events = new EventSource("/api/events");
events.onopen = () => setConnection('connected', '实时连接');
events.onmessage = event => {
  render(JSON.parse(event.data));
  setConnection('connected', '实时连接');
};
events.onerror = () => setConnection('error', '正在重连');
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    world: CoinRunWorld
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.address_string()} {fmt % args}", flush=True)

    def send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def send_event_stream(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        self.world.register_subscriber()
        version = -1
        try:
            while True:
                payload = self.world.wait_for_update(version, timeout=15.0)
                if payload is None:
                    self.wfile.write(b": keepalive\n\n")
                else:
                    version = int(payload["version"])
                    encoded = json.dumps(
                        payload, separators=(",", ":")
                    ).encode("utf-8")
                    self.wfile.write(
                        f"id: {version}\n".encode("ascii")
                        + b"data: "
                        + encoded
                        + b"\n\n"
                    )
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.world.unregister_subscriber()

    def do_GET(self) -> None:
        if self.path == "/":
            data = HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if self.path == "/health":
            state = self.world.snapshot()
            self.send_json(
                {
                    "ok": state["error"] is None,
                    "model": state["model"],
                    "protocol": state["protocol"],
                    "status": state["status"],
                    "streaming": state["streaming"],
                    "step": state["step"],
                    "error": state["error"],
                }
            )
            return
        if self.path == "/api/state":
            self.send_json(self.world.snapshot())
            return
        if self.path == "/api/events":
            self.send_event_stream()
            return
        self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(content_length) or b"{}")
            if self.path == "/api/input":
                pressed = body.get("pressed")
                revision = body.get("revision")
                if not isinstance(pressed, list) or not all(
                    isinstance(value, str) for value in pressed
                ):
                    raise ValueError("pressed must be a list of input names")
                if not isinstance(revision, int):
                    raise ValueError("revision must be an integer")
                payload = self.world.set_input(set(pressed), revision)
            elif self.path == "/api/control":
                paused = body.get("paused")
                if not isinstance(paused, bool):
                    raise ValueError("paused must be a boolean")
                payload = self.world.set_paused(paused)
            elif self.path == "/api/reset":
                payload = self.world.request_reset()
            else:
                self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            self.send_json(payload)
        except Exception as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)


def main() -> None:
    args = parse_args()
    print(f"Loading {args.model_name} from {args.checkpoint}...", flush=True)
    world = CoinRunWorld(
        checkpoint=args.checkpoint,
        dataset=args.dataset,
        model_name=args.model_name,
        context=args.context,
        seed=args.seed,
        denoise_steps=args.denoise_steps,
    )
    Handler.world = world
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    world.start()
    print(
        f"READY http://{args.host}:{args.port} model={args.model_name} "
        f"protocol=continuous-sse-v1 warmup_ms={world.warmup_ms:.1f}",
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
        world.stop()
        world.mesh_context.__exit__(None, None, None)


if __name__ == "__main__":
    main()
