# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import hashlib
import json
import re
import subprocess
import sys
import textwrap
import unittest
from dataclasses import dataclass, field

from torchtitan.config.configurable import Configurable


class TestConfigurable(unittest.TestCase):
    class OldStyleComponent(Configurable):
        """__init__ takes extra runtime kwargs (not config fields)."""

        @dataclass(kw_only=True, slots=True)
        class Config(Configurable.Config):
            x: int = 5

        def __init__(self, config: Config, *, dim: int):
            self.config = config
            self.dim = dim

    class NoKwargsComponent(Configurable):
        """Takes only config, no extra kwargs."""

        @dataclass(kw_only=True, slots=True)
        class Config(Configurable.Config):
            x: int = 5

        def __init__(self, config: Config):
            self.config = config

    def test_valid_config(self):
        """Config with kw_only=True and slots=True should work."""

        class MyComponent(Configurable):
            @dataclass(kw_only=True, slots=True)
            class Config(Configurable.Config):
                x: int = 5

            def __init__(self, config: Config):
                self.config = config

        cfg = MyComponent.Config(x=10)
        obj = cfg.build()
        self.assertIsInstance(obj, MyComponent)
        self.assertEqual(obj.config.x, 10)

    def test_missing_slots_raises(self):
        """Config without slots=True must be rejected."""
        with self.assertRaises(TypeError):

            class BadSlots(Configurable):
                @dataclass(kw_only=True)
                class Config(Configurable.Config):
                    x: int = 5

    def test_missing_kw_only_raises(self):
        """Config without kw_only=True must be rejected."""
        with self.assertRaises(TypeError):

            class BadKwOnly(Configurable):
                @dataclass(slots=True)
                class Config(Configurable.Config):
                    x: int = 5

    def test_missing_both_raises(self):
        """Config without kw_only=True or slots=True must be rejected."""
        with self.assertRaises(TypeError):

            class BadBoth(Configurable):
                @dataclass
                class Config(Configurable.Config):
                    x: int = 5

    def test_build_without_owner_raises(self):
        """Calling build() on the base Config should raise NotImplementedError."""
        cfg = Configurable.Config()
        with self.assertRaises(NotImplementedError):
            cfg.build()

    def test_old_style_forwarding(self):
        """kwargs not in config fields are forwarded to __init__."""
        cfg = self.OldStyleComponent.Config(x=10)
        obj = cfg.build(dim=64)
        self.assertIsInstance(obj, self.OldStyleComponent)
        self.assertEqual(obj.config.x, 10)
        self.assertEqual(obj.dim, 64)

    def test_clone_isolation_old_style(self):
        """Original config is not mutated in old-style path."""
        cfg = self.OldStyleComponent.Config(x=10)
        obj = cfg.build(dim=64)
        obj.config.x = 999
        self.assertEqual(cfg.x, 10)

    def test_no_kwargs(self):
        """build() with no kwargs clones config and constructs."""
        cfg = self.NoKwargsComponent.Config(x=42)
        obj = cfg.build()
        self.assertIsInstance(obj, self.NoKwargsComponent)
        self.assertEqual(obj.config.x, 42)

    def test_no_kwargs_clone_isolation(self):
        """build() with no kwargs still clones the config."""
        cfg = self.NoKwargsComponent.Config(x=42)
        obj = cfg.build()
        obj.config.x = 999
        self.assertEqual(cfg.x, 42)

    def test_to_dict_two_layer(self):
        """to_dict serializes nested configs (two layers deep)."""

        class Inner(Configurable):
            @dataclass(kw_only=True, slots=True)
            class Config(Configurable.Config):
                a: int = 1
                b: int = 2

            def __init__(self, config: Config):
                self.config = config

        class Outer(Configurable):
            @dataclass(kw_only=True, slots=True)
            class Config(Configurable.Config):
                x: int = 10
                inner: Inner.Config = field(default_factory=Inner.Config)

            def __init__(self, config: Config):
                self.config = config

        cfg = Outer.Config(x=42)
        d = cfg.to_dict()
        self.assertEqual(d["x"], 42)
        # Inner config is serialised via its own to_dict
        self.assertIn("inner", d)
        self.assertEqual(d["inner"]["a"], 1)
        self.assertEqual(d["inner"]["b"], 2)

        # After build: all fields present
        obj = cfg.build()
        d2 = obj.config.to_dict()
        self.assertEqual(d2["x"], 42)
        self.assertEqual(d2["inner"]["a"], 1)
        self.assertEqual(d2["inner"]["b"], 2)

    def test_to_dict_is_deterministic_across_fresh_config_processes(self):
        """Equivalent registry configs produce the same address-free snapshot."""
        script = textwrap.dedent(
            """
            import hashlib
            import json

            from torchtitan.config import ConfigManager

            config = ConfigManager().parse_args(
                ["--module", "llama3", "--config", "llama3_debugmodel"]
            )
            canonical = json.dumps(
                config.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            print(json.dumps({
                "canonical": canonical,
                "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            }, sort_keys=True, separators=(",", ":")))
            """
        )

        snapshots = [
            subprocess.run(
                [sys.executable, "-c", script],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            for _ in range(2)
        ]
        first = json.loads(snapshots[0])
        second = json.loads(snapshots[1])

        self.assertEqual(snapshots[0], snapshots[1])
        self.assertEqual(first["canonical"], second["canonical"])
        self.assertEqual(first["sha256"], second["sha256"])
        self.assertEqual(
            first["sha256"],
            hashlib.sha256(first["canonical"].encode("utf-8")).hexdigest(),
        )
        self.assertIsNone(re.search(r"\bat 0x[0-9a-fA-F]+", first["canonical"]))

    def test_to_dict_distinguishes_partial_arguments_and_closure_values(self):
        """Callable snapshots retain values that change callable semantics."""

        class CallableComponent(Configurable):
            @dataclass(kw_only=True, slots=True)
            class Config(Configurable.Config):
                fn: object

            def __init__(self, config: Config):
                self.config = config

        def make_offset(offset):
            def apply_offset(value):
                return value + offset

            return apply_offset

        square = CallableComponent.Config(fn=functools.partial(pow, exp=2)).to_dict()[
            "fn"
        ]
        cube = CallableComponent.Config(fn=functools.partial(pow, exp=3)).to_dict()[
            "fn"
        ]
        offset_one = CallableComponent.Config(fn=make_offset(1)).to_dict()["fn"]
        offset_two = CallableComponent.Config(fn=make_offset(2)).to_dict()["fn"]

        self.assertNotEqual(square, cube)
        self.assertEqual(square["keywords"], {"exp": 2})
        self.assertEqual(cube["keywords"], {"exp": 3})
        self.assertEqual(square["function"]["path"], "builtins.pow")
        self.assertNotEqual(offset_one, offset_two)
        self.assertEqual(offset_one["closure"], {"offset": 1})
        self.assertEqual(offset_two["closure"], {"offset": 2})

    def test_to_dict_distinguishes_function_code_with_the_same_qualified_name(self):
        """Same-qualified functions retain code differences in snapshots."""

        class CallableComponent(Configurable):
            @dataclass(kw_only=True, slots=True)
            class Config(Configurable.Config):
                fn: object

            def __init__(self, config: Config):
                self.config = config

        add_one = lambda value: value + 1
        add_two = lambda value: value + 2

        first = CallableComponent.Config(fn=add_one).to_dict()["fn"]
        second = CallableComponent.Config(fn=add_two).to_dict()["fn"]

        self.assertEqual(first["path"], second["path"])
        self.assertNotEqual(first["code_sha256"], second["code_sha256"])

    def test_to_dict_function_code_identity_ignores_source_location(self):
        """Callable fingerprints do not depend on checkout path or line number."""

        class CallableComponent(Configurable):
            @dataclass(kw_only=True, slots=True)
            class Config(Configurable.Config):
                fn: object

            def __init__(self, config: Config):
                self.config = config

        def load_function(filename, line_padding):
            namespace = {"__name__": "config_callable_fixture"}
            source = (
                "\n" * line_padding + "def configured(value):\n    return value + 1\n"
            )
            exec(compile(source, filename, "exec"), namespace)
            return namespace["configured"]

        first = CallableComponent.Config(
            fn=load_function("/checkout/one/config.py", 0)
        ).to_dict()["fn"]
        second = CallableComponent.Config(
            fn=load_function("/different/checkout/config.py", 9)
        ).to_dict()["fn"]

        self.assertEqual(first["path"], second["path"])
        self.assertEqual(first["code_sha256"], second["code_sha256"])

    def test_to_dict_rejects_opaque_callable_objects(self):
        """Opaque callable state cannot be silently conflated in snapshots."""

        class CallableComponent(Configurable):
            @dataclass(kw_only=True, slots=True)
            class Config(Configurable.Config):
                fn: object

            def __init__(self, config: Config):
                self.config = config

        class StatefulCallable:
            def __init__(self, offset):
                self.offset = offset

            def __call__(self, value):
                return value + self.offset

        with self.assertRaisesRegex(TypeError, "opaque callable.*StatefulCallable"):
            CallableComponent.Config(fn=StatefulCallable(1)).to_dict()

    def test_to_dict_serializes_callable_types_by_qualified_name(self):
        """Callable classes have a stable identity without instance state."""

        class CallableComponent(Configurable):
            @dataclass(kw_only=True, slots=True)
            class Config(Configurable.Config):
                fn: object

            def __init__(self, config: Config):
                self.config = config

        snapshot = CallableComponent.Config(fn=str).to_dict()["fn"]

        self.assertEqual(
            snapshot,
            {"__callable__": "type", "path": "builtins.str"},
        )

    def test_to_dict_rejects_instance_bound_methods(self):
        """Instance-bound methods cannot omit the state of their receiver."""

        class CallableComponent(Configurable):
            @dataclass(kw_only=True, slots=True)
            class Config(Configurable.Config):
                fn: object

            def __init__(self, config: Config):
                self.config = config

        class StatefulCallable:
            def __init__(self, offset):
                self.offset = offset

            def apply(self, value):
                return value + self.offset

        with self.assertRaisesRegex(TypeError, "instance-bound method"):
            CallableComponent.Config(fn=StatefulCallable(1).apply).to_dict()

    def test_to_dict_rejects_instance_bound_builtin_methods(self):
        """Bound builtins cannot omit the state of their receiver."""

        class CallableComponent(Configurable):
            @dataclass(kw_only=True, slots=True)
            class Config(Configurable.Config):
                fn: object

            def __init__(self, config: Config):
                self.config = config

        with self.assertRaisesRegex(TypeError, "instance-bound builtin"):
            CallableComponent.Config(fn={"offset": 1}.get).to_dict()

    def test_to_dict_rejects_unsupported_non_callable_values(self):
        """Unsupported values cannot leak address-bearing repr strings."""

        class UnsupportedValue:
            pass

        class ValueComponent(Configurable):
            @dataclass(kw_only=True, slots=True)
            class Config(Configurable.Config):
                value: object

            def __init__(self, config: Config):
                self.config = config

        with self.assertRaisesRegex(TypeError, "UnsupportedValue"):
            ValueComponent.Config(value=UnsupportedValue()).to_dict()

    def test_traverse_recurse_descends_into_matching_configs(self):
        """traverse(..., recurse=True) descends after yielding matches."""

        class Leaf(Configurable):
            @dataclass(kw_only=True, slots=True)
            class Config(Configurable.Config):
                value: int = 1

            def __init__(self, config: Config):
                self.config = config

        class Middle(Configurable):
            @dataclass(kw_only=True, slots=True)
            class Config(Configurable.Config):
                leaf: Leaf.Config = field(default_factory=Leaf.Config)

            def __init__(self, config: Config):
                self.config = config

        class Outer(Configurable):
            @dataclass(kw_only=True, slots=True)
            class Config(Configurable.Config):
                middle: Middle.Config = field(default_factory=Middle.Config)
                layers: list[Middle.Config] = field(
                    default_factory=lambda: [Middle.Config(), Middle.Config()]
                )

            def __init__(self, config: Config):
                self.config = config

        cfg = Outer.Config()
        self.assertEqual(
            [fqn for fqn, *_ in cfg.traverse(Configurable.Config)],
            [""],
        )
        self.assertEqual(
            [fqn for fqn, *_ in cfg.traverse(Configurable.Config, recurse=True)],
            [
                "",
                "middle",
                "middle.leaf",
                "layers.0",
                "layers.0.leaf",
                "layers.1",
                "layers.1.leaf",
            ],
        )
        recursive = list(cfg.traverse(Configurable.Config, recurse=True))
        by_fqn = {fqn: (parent, attr) for fqn, _, parent, attr in recursive}
        self.assertEqual(by_fqn[""], (None, None))
        self.assertEqual(by_fqn["middle"], (cfg, "middle"))
        self.assertEqual(by_fqn["middle.leaf"], (cfg.middle, "leaf"))
        self.assertEqual(by_fqn["layers.0"], (cfg.layers, 0))
        self.assertEqual(by_fqn["layers.0.leaf"], (cfg.layers[0], "leaf"))

    def test_traverse_includes_root_config(self):
        """traverse(...) yields the root as non-replaceable."""

        class Inner(Configurable):
            @dataclass(kw_only=True, slots=True)
            class Config(Configurable.Config):
                value: int = 1

            def __init__(self, config: Config):
                self.config = config

        class Outer(Configurable):
            @dataclass(kw_only=True, slots=True)
            class Config(Configurable.Config):
                inner: Inner.Config = field(default_factory=Inner.Config)

            def __init__(self, config: Config):
                self.config = config

        cfg = Outer.Config()

        self.assertEqual(
            [
                (fqn, parent, attr)
                for fqn, _cfg, parent, attr in cfg.traverse(Configurable.Config)
            ],
            [("", None, None)],
        )
        self.assertEqual(
            [fqn for fqn, *_ in cfg.traverse(Configurable.Config, recurse=True)],
            ["", "inner"],
        )

    def test_repr(self):
        """repr() works for configs."""
        cfg = self.NoKwargsComponent.Config(x=42)
        r = repr(cfg)
        self.assertIn("x=42", r)


if __name__ == "__main__":
    unittest.main()
