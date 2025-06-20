import os
import torch
import glob
from setuptools import find_packages, setup  # type: ignore

from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    BuildExtension,
    CUDA_HOME,
)


def get_extensions():
    use_cuda = torch.cuda.is_available() and CUDA_HOME is not None
    extension = CUDAExtension if use_cuda else CppExtension

    extra_link_args = []
    extra_compile_args = {
        "cxx": [
            "-O3",
            "-fdiagnostics-color=always",
            "-DPy_LIMITED_API=0x03090000",  # min CPython version 3.9
        ],
        "nvcc": ["-O3"],
    }

    this_dir = os.path.dirname(os.path.curdir)
    rwkv_extensions_dir = os.path.join(this_dir, "rwkv", "model", "csrc")
    rwkv_sources = list(glob.glob(os.path.join(rwkv_extensions_dir, "*.cpp")))

    rwkv_extensions_cuda_dir = os.path.join(rwkv_extensions_dir, "cuda")
    cuda_sources = list(glob.glob(os.path.join(rwkv_extensions_cuda_dir, "*.cu")))

    if use_cuda:
        rwkv_sources += cuda_sources

    fsrs_extensions_dir = os.path.join(this_dir, "fsrs_cpp", "csrc")
    fsrs_sources = list(glob.glob(os.path.join(fsrs_extensions_dir, "*.cpp")))

    ext_modules = [
        # extension(
        #     "rwkv.model.RWKV_CUDA",
        #     rwkv_sources,
        #     extra_compile_args=extra_compile_args,
        #     extra_link_args=extra_link_args,
        #     py_limited_api=False,
        # ),
        extension(
            "fsrs_cpp._FSRS_CPP",
            fsrs_sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            py_limited_api=False,
        )
    ]

    return ext_modules


setup(
    name="srs-benchmark",
    packages=find_packages(),
    ext_modules=get_extensions(),
    install_requires=[
        "torch",
        "tqdm",
        "lmdb",
        "tomli",
        "pandas",
        "pyarrow",
        "fastparquet",
        "wandb",
        "scikit-learn",
    ],
    cmdclass={"build_ext": BuildExtension},
    options={},
)
