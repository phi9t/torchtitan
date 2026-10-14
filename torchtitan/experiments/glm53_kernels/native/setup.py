from __future__ import annotations

import os
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


ROOT = Path(__file__).parent


def _split_flags(value: str) -> list[str]:
    return [item for item in value.split() if item]


extra_compile_args = {
    "cxx": ["-O3", "-std=c++20"],
    "nvcc": [
        "-O3",
        "-std=c++20",
        "--use_fast_math",
        *(_split_flags(os.environ.get("GLM53_NATIVE_NVCC_FLAGS", ""))),
    ],
}

setup(
    name="glm53-kernel-examples",
    version="0.1.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    ext_modules=[
        CppExtension(
            name="glm53_kernel_examples._C",
            sources=[
                "src/glm53_kernel_examples/csrc/scale_add.cpp",
            ],
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
    zip_safe=False,
)

