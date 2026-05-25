import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

from setuptools._distutils import log
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME


ROOT = Path(__file__).resolve().parent
ENZYME_CUDA_SOURCE = ROOT / "parallel" / "csrc" / "enzyme_torch_sample.cu"
FSRS_CUDA_DIR = ROOT / "parallel" / "csrc" / "fsrs"
ENZYME_CUDA_DEPENDENCIES = [ENZYME_CUDA_SOURCE, *sorted(FSRS_CUDA_DIR.glob("*.*"))]
ENZYME_EXTENSION_NAME = "parallel._enzyme_torch_sample"
ENZYME_BUILD_VERBOSE = os.environ.get("ENZYME_BUILD_VERBOSE") == "1"
ENZYME_PLUGIN = Path(
    os.environ.get("ENZYME_CLANG_PLUGIN", "/opt/enzyme/lib/ClangEnzyme-18.so")
)

if not ENZYME_BUILD_VERBOSE:
    if "-q" not in sys.argv and "--quiet" not in sys.argv:
        sys.argv.insert(1, "-q")
    log.set_threshold(log.WARN)


@contextmanager
def quiet_success_output():
    if ENZYME_BUILD_VERBOSE:
        yield
        return

    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)

    with tempfile.TemporaryFile() as stdout_capture, tempfile.TemporaryFile() as stderr_capture:
        restored = False

        def restore() -> None:
            nonlocal restored
            if restored:
                return
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(stdout_fd, 1)
            os.dup2(stderr_fd, 2)
            restored = True

        try:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(stdout_capture.fileno(), 1)
            os.dup2(stderr_capture.fileno(), 2)
            yield
        except BaseException:
            restore()
            stdout_capture.seek(0)
            stderr_capture.seek(0)
            stdout_text = stdout_capture.read().decode(errors="replace")
            stderr_text = stderr_capture.read().decode(errors="replace")
            if stdout_text:
                print(stdout_text, end="")
            if stderr_text:
                print(stderr_text, end="", file=sys.stderr)
            raise
        finally:
            restore()
            os.close(stdout_fd)
            os.close(stderr_fd)


class EnzymeBuildExtension(BuildExtension):
    def build_extension(self, extension) -> None:
        if extension.name != ENZYME_EXTENSION_NAME:
            super().build_extension(extension)
            return

        object_path = Path(self.build_temp) / "enzyme_torch_sample.o"
        self._add_enzyme_object(extension, object_path)
        self._add_enzyme_dependency(extension)

        sources = [Path(source) for source in sorted(extension.sources)]
        source_objects = [
            Path(object_name)
            for object_name in self.compiler.object_filenames(
                [str(source) for source in sources],
                strip_dir=False,
                output_dir=self.build_temp,
            )
        ]
        extension_path = Path(self.get_ext_fullpath(extension.name))

        needs_link = self.force or self._any_newer([*sources, *ENZYME_CUDA_DEPENDENCIES], extension_path)
        if not needs_link:
            log.debug("skipping '%s' extension (up-to-date)", extension.name)
            return

        if self.force or self._any_newer(ENZYME_CUDA_DEPENDENCIES, object_path):
            self._compile_enzyme_cuda_object(object_path)

        needs_source_compile = self.force or any(
            self._source_newer(source, object_path)
            for source, object_path in zip(sources, source_objects, strict=True)
        )
        if needs_source_compile:
            with quiet_success_output():
                super().build_extension(extension)
            return

        log.debug("linking '%s' extension", extension.name)
        self._link_extension(extension, source_objects, extension_path, sources)

    @staticmethod
    def _source_newer(source_path: Path, output_path: Path) -> bool:
        return not output_path.exists() or source_path.stat().st_mtime > output_path.stat().st_mtime

    @classmethod
    def _any_newer(cls, source_paths: list[Path], output_path: Path) -> bool:
        return not output_path.exists() or any(
            cls._source_newer(source_path, output_path) for source_path in source_paths
        )

    @staticmethod
    def _add_enzyme_object(extension, object_path: Path) -> None:
        extra_objects = list(getattr(extension, "extra_objects", None) or [])
        object_path_s = str(object_path)
        if object_path_s not in extra_objects:
            extra_objects.append(object_path_s)
        extension.extra_objects = extra_objects

    @staticmethod
    def _add_enzyme_dependency(extension) -> None:
        depends = list(getattr(extension, "depends", None) or [])
        for source_path in ENZYME_CUDA_DEPENDENCIES:
            source_path_s = str(source_path)
            if source_path_s not in depends:
                depends.append(source_path_s)
        extension.depends = depends

    def _compile_enzyme_cuda_object(self, object_path: Path) -> None:
        object_path.parent.mkdir(parents=True, exist_ok=True)
        if not ENZYME_CUDA_SOURCE.exists():
            raise FileNotFoundError(f"missing Enzyme CUDA source: {ENZYME_CUDA_SOURCE}")
        if not ENZYME_PLUGIN.exists():
            raise FileNotFoundError(f"missing Enzyme Clang plugin: {ENZYME_PLUGIN}")

        clang = os.environ.get("ENZYME_CLANG", "clang")
        cuda_home = os.environ.get("CUDA_HOME") or CUDA_HOME or "/usr/local/cuda"
        cuda_arch = os.environ.get("ENZYME_CUDA_ARCH", "sm_70")

        command = [
            clang,
            "-x",
            "cuda",
            str(ENZYME_CUDA_SOURCE),
            "-c",
            "-o",
            str(object_path),
            "-O2",
            "-fPIC",
            "-std=c++17",
            f"-fplugin={ENZYME_PLUGIN}",
            f"--cuda-gpu-arch={cuda_arch}",
            f"--cuda-path={cuda_home}",
            "-I",
            str(Path(cuda_home) / "include"),
        ]

        run_kwargs = {"check": True}
        if not ENZYME_BUILD_VERBOSE:
            run_kwargs["stdout"] = subprocess.DEVNULL
        subprocess.run(command, **run_kwargs)

    def _link_extension(
        self,
        extension,
        source_objects: list[Path],
        extension_path: Path,
        sources: list[Path],
    ) -> None:
        self.mkpath(str(extension_path.parent))
        objects = [str(object_path) for object_path in source_objects]
        objects.extend(str(object_path) for object_path in extension.extra_objects)
        extra_args = extension.extra_link_args or []
        language = extension.language or self.compiler.detect_language([str(source) for source in sources])

        with quiet_success_output():
            self.compiler.link_shared_object(
                objects,
                str(extension_path),
                libraries=self.get_libraries(extension),
                library_dirs=extension.library_dirs,
                runtime_library_dirs=extension.runtime_library_dirs,
                extra_postargs=extra_args,
                export_symbols=self.get_export_symbols(extension),
                debug=self.debug,
                build_temp=self.build_temp,
                target_lang=language,
            )


setup(
    name="srs-benchmark-enzyme-torch-sample",
    packages=[],
    ext_modules=[
        CUDAExtension(
            name="parallel._enzyme_torch_sample",
            sources=["parallel/csrc/enzyme_torch_sample.cpp"],
            extra_compile_args={
                "cxx": ["-O2", "-std=c++17"],
            },
        ),
    ],
    cmdclass={"build_ext": EnzymeBuildExtension},
)
