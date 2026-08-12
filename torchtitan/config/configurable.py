# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import functools
import hashlib
import json
import logging
import types
from collections.abc import Iterator
from dataclasses import dataclass, fields, replace
from typing import ClassVar

from torchtitan.observability import structured_logger as sl

logger = logging.getLogger(__name__)


def _qualified_name(value: object) -> str:
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if module is None or qualname is None:
        value_type = type(value)
        module = value_type.__module__
        qualname = value_type.__qualname__
    return f"{module}.{qualname}" if module else qualname


def _code_constant_to_dict(value: object):
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return {"type": "float", "value": value.hex()}
    if isinstance(value, complex):
        return {
            "type": "complex",
            "real": value.real.hex(),
            "imag": value.imag.hex(),
        }
    if isinstance(value, bytes):
        return {"type": "bytes", "value": value.hex()}
    if isinstance(value, tuple):
        return {
            "type": "tuple",
            "items": [_code_constant_to_dict(item) for item in value],
        }
    if isinstance(value, frozenset):
        items = [_code_constant_to_dict(item) for item in value]
        items.sort(
            key=lambda item: json.dumps(
                item, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
        )
        return {"type": "frozenset", "items": items}
    if isinstance(value, types.CodeType):
        return {"type": "code", "value": _code_to_dict(value)}
    if value is Ellipsis:
        return {"type": "ellipsis"}
    raise TypeError(
        "cannot fingerprint function code constant of type "
        f"{_qualified_name(type(value))}"
    )


def _code_to_dict(code: types.CodeType) -> dict:
    """Return code semantics without filenames, lines, or debug tables."""
    return {
        "argcount": code.co_argcount,
        "posonlyargcount": code.co_posonlyargcount,
        "kwonlyargcount": code.co_kwonlyargcount,
        "nlocals": code.co_nlocals,
        "stacksize": code.co_stacksize,
        "flags": code.co_flags,
        "bytecode": code.co_code.hex(),
        "constants": [_code_constant_to_dict(item) for item in code.co_consts],
        "names": list(code.co_names),
        "varnames": list(code.co_varnames),
        "freevars": list(code.co_freevars),
        "cellvars": list(code.co_cellvars),
        "name": code.co_name,
        "qualname": getattr(code, "co_qualname", code.co_name),
        "exceptiontable": getattr(code, "co_exceptiontable", b"").hex(),
    }


def _code_sha256(code: types.CodeType) -> str:
    payload = json.dumps(
        _code_to_dict(code),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _callable_to_dict(value: object) -> dict[str, object]:
    if isinstance(value, functools.partial):
        return {
            "__callable__": "partial",
            "function": _callable_to_dict(value.func),
            "args": _config_value_to_dict(value.args),
            "keywords": _config_value_to_dict(value.keywords or {}),
        }

    if isinstance(value, types.FunctionType):
        result: dict[str, object] = {
            "__callable__": "function",
            "path": _qualified_name(value),
            "code_sha256": _code_sha256(value.__code__),
        }
        if value.__defaults__:
            result["defaults"] = _config_value_to_dict(value.__defaults__)
        if value.__kwdefaults__:
            result["keyword_defaults"] = _config_value_to_dict(value.__kwdefaults__)
        if value.__closure__:
            closure: dict[str, object] = {}
            for name, cell in zip(value.__code__.co_freevars, value.__closure__):
                try:
                    cell_value = cell.cell_contents
                except ValueError:
                    cell_value = {"__empty_cell__": True}
                closure[name] = _config_value_to_dict(cell_value)
            result["closure"] = closure
        return result

    if isinstance(value, (types.BuiltinFunctionType, types.BuiltinMethodType)):
        owner = getattr(value, "__self__", None)
        if owner is not None and not isinstance(owner, types.ModuleType):
            raise TypeError(
                "cannot serialize instance-bound builtin config value "
                f"{_qualified_name(value)} without conflating receiver state"
            )
        return {
            "__callable__": "builtin",
            "path": _qualified_name(value),
        }

    if isinstance(value, types.MethodType) and not isinstance(value.__self__, type):
        raise TypeError(
            "cannot serialize instance-bound method config value "
            f"{_qualified_name(value)} without conflating receiver state"
        )

    if isinstance(value, type):
        return {
            "__callable__": "type",
            "path": _qualified_name(value),
        }

    raise TypeError(
        "cannot serialize opaque callable config value of type "
        f"{_qualified_name(type(value))}; use a function, builtin, partial, "
        "bound method, or type"
    )


def _config_value_to_dict(value: object):
    if hasattr(value, "to_dict"):
        return _config_value_to_dict(value.to_dict())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _config_value_to_dict(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, (list, tuple)):
        return type(value)(_config_value_to_dict(item) for item in value)
    if isinstance(value, dict):
        return {key: _config_value_to_dict(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if callable(value):
        return _callable_to_dict(value)

    raise TypeError(
        "cannot serialize config field value of unsupported type "
        f"{_qualified_name(type(value))}"
    )


class Configurable:
    """Base class for all configurable components.

    Every configurable class:
    - Inherits from Configurable (or Module for nn.Module components)
    - Defines a nested Config(Configurable.Config) with @dataclass(kw_only=True, slots=True)
    - Gets build() auto-wired via __init_subclass__ (no manual override needed)
    - Accepts ``__init__(self, config: Config)``

    build() has two modes:
    - No kwargs: ``self._owner(config=replace(self))``
    - With kwargs (runtime objects not in config): forwarded to
      ``self._owner(config=..., **kwargs)``.  Used by non-model
      Configurables (tokenizer, dataloader, optimizer, etc.)
      that receive runtime objects at construction time.

    Enforcement: Configurable.__init_subclass__ checks that every Config uses
    @dataclass(kw_only=True, slots=True). This check runs on the OUTER class
    (not Config.__init_subclass__) because @dataclass(slots=True) replaces the
    class, so Config.__init_subclass__ sees the pre-decorator version.
    """

    @dataclass(kw_only=True, slots=True)
    class Config:
        """Base config class for all configurable components."""

        _owner: ClassVar[type | None] = None

        def to_dict(self) -> dict:
            """Serialize config to a deterministic, JSON-compatible plain dict.

            Callable identities include their import path, partial arguments, and
            closure values. Opaque callable instances have no general inspectable
            state contract and are rejected instead of conflating their semantics.
            """

            return {
                f.name: _config_value_to_dict(getattr(self, f.name))
                for f in fields(self)
                if not f.name.startswith("_")
            }

        def traverse(
            self, config_cls: type, *, recurse: bool = False, _prefix: str = ""
        ) -> Iterator[
            tuple[str, "Configurable.Config", object | None, str | int | None]
        ]:
            """Yield ``(fqn, config, parent, field_name)`` for every nested config of *config_cls*.

            Recursively traverses dataclass fields, including items inside lists.
            The *fqn* mirrors the module FQN that ``build()`` would produce
            (e.g. ``"layers.0.feed_forward.w1"``).

            *parent* and *field_name* allow replacing the config in the tree.
            They are ``None`` for the root config::

                for fqn, cfg, parent, attr in model_config.traverse(Linear.Config):
                    if parent is None:
                        model_config = NewConfig(...)
                    else:
                        setattr(parent, attr, NewConfig(...))

            When ``recurse`` is ``True``, traversal continues into matching
            configs after yielding them.  The default preserves the historical
            behavior where matching a config stops descent into that subtree.
            """
            if isinstance(self, config_cls):
                yield _prefix, self, None, None
                if not recurse:
                    return

            def _traverse_child(val, prefix, parent, attr):
                for child_fqn, child_config, child_parent, child_attr in val.traverse(
                    config_cls, recurse=recurse, _prefix=prefix
                ):
                    if child_parent is None:
                        yield child_fqn, child_config, parent, attr
                    else:
                        yield child_fqn, child_config, child_parent, child_attr

            for f in fields(self):
                val = getattr(self, f.name)
                fqn = f"{_prefix}.{f.name}" if _prefix else f.name
                if isinstance(val, config_cls):
                    if recurse and hasattr(val, "traverse"):
                        yield from _traverse_child(val, fqn, self, f.name)
                    else:
                        yield fqn, val, self, f.name
                elif isinstance(val, list):
                    for i, item in enumerate(val):
                        item_fqn = f"{fqn}.{i}"
                        if isinstance(item, config_cls):
                            if recurse and hasattr(item, "traverse"):
                                yield from _traverse_child(item, item_fqn, val, i)
                            else:
                                yield item_fqn, item, val, i
                        elif hasattr(item, "traverse"):
                            yield from _traverse_child(item, item_fqn, val, i)
                elif hasattr(val, "traverse"):
                    yield from _traverse_child(val, fqn, self, f.name)

        def build(self, **kwargs):
            """Construct the owning class. Auto-wired by __init_subclass__.

            Two modes:
            - No kwargs: ``self._owner(config=replace(self))``
            - With kwargs (runtime objects not in config): forwarded to
              ``self._owner(config=..., **kwargs)``.  Used by non-model
              Configurables (tokenizer, dataloader, optimizer, etc.)
              that receive runtime objects at construction time.
            """
            if self._owner is None:
                raise NotImplementedError(
                    f"{type(self).__name__} has no owner class. "
                    "Define Config inside a Configurable subclass."
                )
            with sl.log_trace_span(f"{type(self).__qualname__}.build"):
                if not kwargs:
                    return self._owner(config=replace(self))

                config_fields = {f.name for f in fields(self)}
                overlap = config_fields & kwargs.keys()
                if overlap:
                    raise ValueError(
                        f"build() kwargs {overlap} overlap with config fields. "
                        "Put these values in the Config, not in build() kwargs."
                    )
                return self._owner(config=replace(self), **kwargs)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "Config" in cls.__dict__:
            config_cls = cls.__dict__["Config"]
            if issubclass(config_cls, Configurable.Config):
                # Enforce @dataclass(kw_only=True, slots=True)
                if "__slots__" not in config_cls.__dict__:
                    raise TypeError(
                        f"{cls.__name__}.Config must use "
                        "@dataclass(kw_only=True, slots=True)"
                    )
                for f in fields(config_cls):
                    if f.init and not f.kw_only:
                        raise TypeError(
                            f"{cls.__name__}.Config field '{f.name}' "
                            "must be keyword-only"
                        )
                # Auto-wire build() to construct this class
                config_cls._owner = cls
