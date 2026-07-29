#!/usr/bin/env python3
"""Local browser demo for action-conditioned CoinRun world-model inference."""

from __future__ import annotations

import argparse
import base64
import functools
import io
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
from omegaconf import OmegaConf

from dreamer.actions import Actions, shift_actions
from dreamer.checkpointing import DynamicsCheckpointBundle
from dreamer.data import build_iterator
from dreamer.generation import DenoiseSchedule, latent_rollout
from dreamer.parallel import build_parallel
from dreamer.sampler import decode_jit, encode_jit


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
        self.lock = threading.Lock()

        config_dir = Path(__file__).resolve().parent / "configs"
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
        self.step_count = 0
        self.last_action = 4
        self.last_latency_ms = 0.0
        self.reset()
        self.warmup_ms = self._warmup()

    def _warmup(self) -> float:
        saved = (
            self.latents_ctx,
            self.actions_ctx,
            self.rng,
            self.current_frame,
            self.step_count,
            self.last_action,
        )
        started = time.perf_counter()
        self._step_once(4)
        jax.block_until_ready(self.current_frame)
        elapsed = (time.perf_counter() - started) * 1000.0
        (
            self.latents_ctx,
            self.actions_ctx,
            self.rng,
            self.current_frame,
            self.step_count,
            self.last_action,
        ) = saved
        self.last_latency_ms = 0.0
        return elapsed

    def reset(self) -> dict[str, Any]:
        batch = next(self.dataloader)
        frames = batch["videos"]
        self.latents_ctx = encode_jit(self.tokenizer, frames)
        self.actions_ctx = shift_actions(
            batch["actions"], self.cfg.dataset.categorical_action_dim
        )
        self.current_frame = jnp.clip(frames[0, -1], 0, 255).astype(jnp.uint8)
        self.step_count = 0
        self.last_action = 4
        self.last_latency_ms = 0.0
        return self.state()

    def _step_once(self, action_id: int) -> None:
        action = Actions(
            categorical=jnp.asarray([[action_id]], dtype=jnp.int32)
        )
        self.rng, step_rng = jax.random.split(self.rng)
        rollout = latent_rollout(
            self.dynamics,
            actions_future=action,
            schedule=self.schedule,
            latents_ctx=self.latents_ctx,
            actions_ctx=self.actions_ctx,
            num_steps=1,
            rng=step_rng,
        )
        next_latent = rollout["latents"][:, -1:]
        decoded = decode_jit(self.tokenizer, next_latent)
        self.current_frame = jnp.clip(decoded[0, -1], 0, 255).astype(jnp.uint8)
        self.latents_ctx = append_context_latent(self.latents_ctx, next_latent)
        self.actions_ctx = Actions(
            categorical=jnp.concatenate(
                [self.actions_ctx.categorical[:, 1:], action.categorical], axis=1
            )
        )
        self.step_count += 1
        self.last_action = action_id

    def step(self, action_id: int, repeat: int = 1) -> dict[str, Any]:
        if action_id not in ACTION_NAMES:
            raise ValueError(f"Unsupported action {action_id}; choose {sorted(ACTION_NAMES)}")
        repeat = max(1, min(int(repeat), 8))
        started = time.perf_counter()
        for _ in range(repeat):
            self._step_once(action_id)
        jax.block_until_ready(self.current_frame)
        self.last_latency_ms = (time.perf_counter() - started) * 1000.0 / repeat
        return self.state()

    def state(self) -> dict[str, Any]:
        return {
            "frame": png_data_url(self.current_frame),
            "model": self.model_name,
            "context_frames": self.context,
            "denoise_steps": self.schedule.num_steps,
            "step": self.step_count,
            "action_id": self.last_action,
            "action": ACTION_NAMES[self.last_action],
            "latency_ms": round(self.last_latency_ms, 1),
            "warmup_ms": round(getattr(self, "warmup_ms", 0.0), 1),
        }


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>OpenDreamer CoinRun Live Demo</title>
  <style>
    :root { color-scheme: dark; --bg:#090b0f; --panel:#121720; --line:#273041;
      --ink:#f2f5fa; --muted:#96a2b5; --green:#79e6ae; --orange:#ffb86b; }
    * { box-sizing:border-box; }
    body { margin:0; min-height:100vh; background:radial-gradient(circle at 50% -15%,#1e293b 0,#090b0f 45%);
      color:var(--ink); font:15px/1.45 ui-sans-serif,system-ui,-apple-system,sans-serif;
      display:grid; place-items:center; padding:28px; }
    main { width:min(760px,100%); }
    .eyebrow { color:var(--green); font-weight:700; letter-spacing:.12em; text-transform:uppercase; font-size:12px; }
    h1 { margin:8px 0 6px; font-size:clamp(25px,5vw,42px); letter-spacing:-.04em; }
    .sub { color:var(--muted); margin:0 0 20px; }
    .screen { border:1px solid var(--line); border-radius:18px; overflow:hidden; background:#050608;
      box-shadow:0 25px 80px #0009; }
    .screen img { display:block; width:100%; aspect-ratio:1/1; object-fit:contain; image-rendering:pixelated; }
    .stats { display:flex; gap:8px; flex-wrap:wrap; padding:12px; background:var(--panel); border-top:1px solid var(--line); }
    .pill { background:#0c1118; border:1px solid var(--line); border-radius:999px; padding:5px 10px; color:var(--muted); }
    .pill b { color:var(--ink); }
    .controls { margin-top:16px; display:grid; grid-template-columns:repeat(6,1fr); gap:9px; }
    button { min-height:58px; border:1px solid var(--line); border-radius:13px; background:linear-gradient(#1a2230,#111721);
      color:var(--ink); font:inherit; font-weight:700; cursor:pointer; touch-action:none; user-select:none; }
    button:hover { border-color:#52627a; transform:translateY(-1px); }
    button:active,.active { background:#254b3c; border-color:var(--green); transform:translateY(1px); }
    .reset { margin-top:10px; width:100%; min-height:44px; color:var(--orange); }
    .hint { color:var(--muted); font-size:13px; margin-top:13px; text-align:center; }
    .busy { opacity:.65; }
    @media(max-width:640px) { .controls { grid-template-columns:repeat(3,1fr); } body { padding:14px; } }
  </style>
</head>
<body>
<main>
  <div class="eyebrow">Action-conditioned world model</div>
  <h1>CoinRun · 梦里的世界</h1>
  <p class="sub">你不是在操作真实游戏，而是在逐帧推动模型想象出的下一刻。</p>
  <section class="screen">
    <img id="frame" alt="CoinRun predicted frame">
    <div class="stats">
      <span class="pill">模型 <b id="model">—</b></span>
      <span class="pill">历史 <b id="ctx">—</b> 帧</span>
      <span class="pill">步数 <b id="step">0</b></span>
      <span class="pill">动作 <b id="action">—</b></span>
      <span class="pill">推理 <b id="latency">—</b></span>
    </div>
  </section>
  <section class="controls" id="controls">
    <button data-action="1">← 左</button>
    <button data-action="2">↖ 左跳</button>
    <button data-action="4">· 停一步</button>
    <button data-action="5">↑ 跳</button>
    <button data-action="7">右 →</button>
    <button data-action="8">右跳 ↗</button>
  </section>
  <button class="reset" id="reset">换一段真实历史，重新开始</button>
  <p class="hint">键盘：← / → / 空格；方向键和空格同时按就是斜跳。按住按钮可连续生成。</p>
</main>
<script>
const frame = document.querySelector('#frame');
const controls = document.querySelector('#controls');
const keys = new Set();
let pending = false, holdTimer = null, heldButton = null;

function render(s) {
  frame.src = s.frame;
  document.querySelector('#model').textContent = s.model;
  document.querySelector('#ctx').textContent = s.context_frames;
  document.querySelector('#step').textContent = s.step;
  document.querySelector('#action').textContent = s.action;
  document.querySelector('#latency').textContent = s.latency_ms ? `${s.latency_ms} ms/帧` : '已就绪';
}
async function api(path, body) {
  if (pending) return;
  pending = true; controls.classList.add('busy');
  try {
    const r = await fetch(path, {method: body ? 'POST':'GET',
      headers:{'Content-Type':'application/json'}, body:body ? JSON.stringify(body):undefined});
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || r.statusText);
    render(data);
  } catch (e) { alert(e.message); }
  finally { pending=false; controls.classList.remove('busy'); }
}
function actionFromKeys() {
  const left=keys.has('ArrowLeft'), right=keys.has('ArrowRight'), jump=keys.has('Space');
  if (left && jump) return 2;
  if (right && jump) return 8;
  if (left) return 1;
  if (right) return 7;
  if (jump) return 5;
  return null;
}
function beginHold(action, button) {
  heldButton=button; button?.classList.add('active');
  api('/api/step',{action});
  holdTimer=setInterval(()=>api('/api/step',{action}),140);
}
function endHold() {
  clearInterval(holdTimer); holdTimer=null;
  heldButton?.classList.remove('active'); heldButton=null;
}
for (const b of controls.querySelectorAll('button')) {
  b.addEventListener('pointerdown', e => { e.preventDefault(); beginHold(Number(b.dataset.action),b); });
  b.addEventListener('pointerup', endHold);
  b.addEventListener('pointercancel', endHold);
  b.addEventListener('pointerleave', endHold);
}
addEventListener('keydown', e => {
  if (!['ArrowLeft','ArrowRight','Space'].includes(e.code)) return;
  e.preventDefault(); keys.add(e.code);
  if (!holdTimer) beginHold(actionFromKeys(),null);
});
addEventListener('keyup', e => {
  keys.delete(e.code); endHold();
  const next=actionFromKeys(); if(next!==null) beginHold(next,null);
});
document.querySelector('#reset').onclick=()=>api('/api/reset',{});
api('/api/state');
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    world: CoinRunWorld

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
            self.send_json({"ok": True, "model": self.world.model_name})
            return
        if self.path == "/api/state":
            with self.world.lock:
                self.send_json(self.world.state())
            return
        self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(content_length) or b"{}")
            with self.world.lock:
                if self.path == "/api/step":
                    payload = self.world.step(
                        int(body.get("action", 4)), int(body.get("repeat", 1))
                    )
                elif self.path == "/api/reset":
                    payload = self.world.reset()
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
    print(
        f"READY http://{args.host}:{args.port} model={args.model_name} "
        f"warmup_ms={world.warmup_ms:.1f}",
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
        world.mesh_context.__exit__(None, None, None)


if __name__ == "__main__":
    main()
