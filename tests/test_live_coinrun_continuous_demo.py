import ast
import types
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "experiments"
    / "coinrun"
    / "live_coinrun_continuous_demo.py"
)


class FakeArray:
    def __init__(self, dtype: str, shape: tuple[int, ...]) -> None:
        self.dtype = dtype
        self.shape = shape

    def __getitem__(self, key: object) -> "FakeArray":
        del key
        return self

    def astype(self, dtype: str) -> "FakeArray":
        return FakeArray(dtype, self.shape)


class FakeJnp:
    @staticmethod
    def concatenate(arrays: list[FakeArray], axis: int) -> FakeArray:
        assert axis == 1
        dtype = arrays[0].dtype
        if any(array.dtype != dtype for array in arrays):
            dtype = "float32"
        return FakeArray(dtype, arrays[0].shape)


def load_append_context_latent():
    tree = ast.parse(SCRIPT_PATH.read_text())
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "append_context_latent"
    ]
    if len(functions) != 1:
        raise AssertionError("append_context_latent must be defined exactly once")
    module = ast.Module(body=functions, type_ignores=[])
    namespace = {
        "jax": types.SimpleNamespace(Array=FakeArray),
        "jnp": FakeJnp,
    }
    exec(compile(module, str(SCRIPT_PATH), "exec"), namespace)
    return namespace["append_context_latent"]


def load_resolve_config_dir():
    tree = ast.parse(SCRIPT_PATH.read_text())
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "resolve_config_dir"
    ]
    if len(functions) != 1:
        raise AssertionError("resolve_config_dir must be defined exactly once")
    module = ast.Module(body=functions, type_ignores=[])
    namespace = {"Path": Path}
    exec(compile(module, str(SCRIPT_PATH), "exec"), namespace)
    return namespace["resolve_config_dir"]


def load_resolve_pressed_action():
    tree = ast.parse(SCRIPT_PATH.read_text())
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "resolve_pressed_action"
    ]
    if len(functions) != 1:
        raise AssertionError("resolve_pressed_action must be defined exactly once")
    module = ast.Module(body=functions, type_ignores=[])
    namespace: dict[str, object] = {
        "VALID_INPUTS": frozenset({"left", "right", "jump"})
    }
    exec(compile(module, str(SCRIPT_PATH), "exec"), namespace)
    return namespace["resolve_pressed_action"]


def method_call_names(class_name: str, method_name: str) -> set[str]:
    tree = ast.parse(SCRIPT_PATH.read_text())
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    if len(classes) != 1:
        raise AssertionError(f"{class_name} must be defined exactly once")
    methods = [
        node
        for node in classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    ]
    if len(methods) != 1:
        raise AssertionError(
            f"{class_name}.{method_name} must be defined exactly once"
        )
    return {
        node.func.id
        for node in ast.walk(methods[0])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


class LiveCoinRunDemoTests(unittest.TestCase):
    def test_append_context_latent_preserves_context_dtype(self) -> None:
        latents_ctx = FakeArray("bfloat16", (1, 16, 2, 4))
        generated_latent = FakeArray("float32", (1, 1, 2, 4))

        updated = load_append_context_latent()(latents_ctx, generated_latent)

        self.assertEqual(updated.dtype, "bfloat16")
        self.assertEqual(updated.shape, latents_ctx.shape)

    def test_config_dir_resolves_to_repository_root(self) -> None:
        resolved = load_resolve_config_dir()(SCRIPT_PATH)

        self.assertEqual(
            resolved,
            SCRIPT_PATH.parents[3] / "configs",
        )
        self.assertTrue((resolved / "eval_fvd.yaml").is_file())

    def test_pressed_key_state_maps_to_complete_coinrun_action(self) -> None:
        resolve = load_resolve_pressed_action()

        self.assertEqual(resolve(set()), 4)
        self.assertEqual(resolve({"left"}), 1)
        self.assertEqual(resolve({"left", "jump"}), 2)
        self.assertEqual(resolve({"jump"}), 5)
        self.assertEqual(resolve({"right"}), 7)
        self.assertEqual(resolve({"right", "jump"}), 8)
        with self.assertRaisesRegex(ValueError, "Unsupported input"):
            resolve({"down"})

    def test_generated_frame_reuses_persistent_cache_step(self) -> None:
        calls = method_call_names("CoinRunWorld", "_step_once")

        self.assertIn("next_frame_jit", calls)
        self.assertNotIn("latent_rollout", calls)

    def test_browser_uses_continuous_event_stream_and_stateful_input(self) -> None:
        source = SCRIPT_PATH.read_text()

        self.assertIn('new EventSource("/api/events")', source)
        self.assertIn("'/api/input'", source)
        self.assertNotIn("fetch('/api/step'", source)
        self.assertNotIn("setInterval(()=>api('/api/step'", source)

    def test_desktop_layout_keeps_stage_inside_initial_viewport(self) -> None:
        source = SCRIPT_PATH.read_text()

        self.assertIn("height:100dvh", source)
        self.assertIn("max-height:calc(100dvh", source)
        self.assertIn("grid-template-columns:minmax(0,1fr) 240px", source)


if __name__ == "__main__":
    unittest.main()
