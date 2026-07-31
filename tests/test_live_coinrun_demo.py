import ast
import types
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "experiments"
    / "coinrun"
    / "live_coinrun_demo.py"
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


if __name__ == "__main__":
    unittest.main()
