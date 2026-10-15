from __future__ import annotations

import csv
import hashlib
import importlib
import importlib.metadata as importlib_metadata
import json
import os
import site
import sys
from pathlib import Path


MODULES = (
    "torch",
    "triton",
    "fla",
    "flash_attn",
    "attn_gym",
    "torch_remat",
    "spmd_types",
    "torchao",
)
IMPORT_CHECKS = (
    "fla.ops.kda",
    "torchao.prototype.moe_training",
    "torchao.prototype.moe_training.kernels.float8_rowwise",
)
DIST_NAMES = (
    "torch",
    "triton",
    "fla-core",
    "flash-attn",
    "attn-gym",
    "torch_remat",
    "spmd_types",
    "torchao",
)


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def version_of(module: object) -> str | None:
    value = getattr(module, "__version__", None)
    if value is None:
        return None
    return str(value)


def module_info(name: str) -> dict[str, object]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    path = getattr(module, "__file__", None)
    return {
        "ok": True,
        "version": version_of(module),
        "file": path,
        "file_sha256": file_sha256(Path(path)) if path else None,
    }


def import_check(name: str) -> dict[str, object]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    path = getattr(module, "__file__", None)
    return {"ok": True, "file": path}


def distribution_info(name: str) -> dict[str, object]:
    try:
        dist = importlib_metadata.distribution(name)
    except importlib_metadata.PackageNotFoundError:
        return {"ok": False, "error": "not installed"}

    dist_info_path = Path(str(dist._path))  # noqa: SLF001
    record_path = dist_info_path / "RECORD"
    record_entries = []
    if record_path.is_file():
        with record_path.open(newline="") as handle:
            for row in csv.reader(handle):
                if not row:
                    continue
                rel_path = row[0]
                digest = row[1] if len(row) > 1 else ""
                if (
                    rel_path.endswith((".dist-info/METADATA", ".dist-info/WHEEL"))
                    or rel_path.endswith("__init__.py")
                    or "autotun" in rel_path
                    or "float8_rowwise" in rel_path
                    or "chunk_delta_h" in rel_path
                    or "kda" in rel_path
                ):
                    record_entries.append(
                        {
                            "path": rel_path,
                            "hash": digest,
                            "size": row[2] if len(row) > 2 else "",
                        }
                    )

    return {
        "ok": True,
        "version": dist.version,
        "location": str(dist.locate_file("")),
        "dist_info": str(dist_info_path),
        "metadata_sha256": file_sha256(dist_info_path / "METADATA"),
        "wheel_sha256": file_sha256(dist_info_path / "WHEEL"),
        "record_sha256": file_sha256(record_path),
        "record_entry_count": sum(1 for _ in record_path.open())
        if record_path.is_file()
        else None,
        "selected_record_entries": record_entries,
    }


def main() -> int:
    label = os.environ.get("GLM53_ENV_PROBE_LABEL", "unlabeled")
    env_keys = sorted(
        key
        for key in os.environ
        if key.startswith(("TRITON_", "TORCHTITAN_", "PYTHON"))
        or key in {"PATH", "VIRTUAL_ENV", "CUDA_HOME", "CUDA_PATH"}
    )
    report = {
        "label": label,
        "executable": sys.executable,
        "version": sys.version,
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "path": sys.path,
        "site_packages": site.getsitepackages(),
        "user_site": site.getusersitepackages(),
        "pyvenv_cfg": Path(sys.prefix, "pyvenv.cfg").read_text().splitlines()
        if Path(sys.prefix, "pyvenv.cfg").is_file()
        else [],
        "env": {key: os.environ.get(key) for key in env_keys},
        "modules": {name: module_info(name) for name in MODULES},
        "import_checks": {name: import_check(name) for name in IMPORT_CHECKS},
        "distributions": {name: distribution_info(name) for name in DIST_NAMES},
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
