from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import sys
import time


def _requested_device_arg(argv: list[str]) -> str | None:
    for i, arg in enumerate(argv):
        if arg == "--device" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--device="):
            return arg.split("=", 1)[1]
    return None


_device_arg = _requested_device_arg(sys.argv[1:])
if _device_arg == "cpu":
    os.environ["JAX_PLATFORMS"] = "cpu"
elif _device_arg == "cuda":
    os.environ["JAX_PLATFORMS"] = "cuda"


import jax
import torch
from tqdm.auto import tqdm  # type: ignore

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from parallel.models import fsrs_v7_jax
from parallel.models.fsrs_v7_constants import FSRS7_DEFAULT_35_VALUES


@dataclass(frozen=True, order=True)
class Shape:
    batch_size: int
    seq_len: int

    @property
    def elements(self) -> int:
        return self.batch_size * self.seq_len


@dataclass(frozen=True)
class Timing:
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float

    @classmethod
    def from_samples(cls, samples: list[float]) -> Timing:
        return cls(
            mean_ms=statistics.mean(samples),
            median_ms=statistics.median(samples),
            min_ms=min(samples),
            max_ms=max(samples),
        )


@dataclass(frozen=True)
class ShapeResult:
    shape: Shape
    samples: int
    timing: Timing
    rows_per_second: float
    elements_per_second: float
    us_per_row: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure FSRS v7 JAX loss/prediction/grad runtime over a shuffled "
            "(B, L) grid, including torch<->jax conversions."
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--min-batch-exp", type=int, default=14)
    parser.add_argument("--max-batch-exp", type=int, default=22)
    parser.add_argument("--max-seq-len", type=int, default=65)
    parser.add_argument("--min-seq-len", type=int, default=2)
    parser.add_argument("--seq-len-ratio", type=float, default=math.sqrt(2.0))
    parser.add_argument("--min-elements-exp", type=int, default=17)
    parser.add_argument("--max-elements-exp", type=int, default=26)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def choose_torch_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda, but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    if torch.cuda.is_available() and jax.default_backend() in {"cuda", "gpu"}:
        return torch.device("cuda")
    return torch.device("cpu")


def safe_filename_part(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-") or "unknown"


def default_output_path() -> Path:
    device_name = safe_filename_part(str(jax.devices()[0]))
    return Path(".batch-timings") / f"fsrs_v7_jax_shape_grid_{device_name}.json"


def seq_lens_by_ratio(max_seq_len: int, min_seq_len: int, ratio: float) -> list[int]:
    if min_seq_len < 2:
        raise ValueError("--min-seq-len must be at least 2.")
    if max_seq_len < min_seq_len:
        raise ValueError("--max-seq-len must be at least --min-seq-len.")
    if ratio <= 1.0:
        raise ValueError("--seq-len-ratio must be greater than 1.")

    lens = [max_seq_len]
    current = max_seq_len
    while current > min_seq_len:
        next_len = max(min_seq_len, int(round(current / ratio)))
        if next_len >= current:
            next_len = current - 1
        next_len = max(min_seq_len, next_len)
        if next_len != lens[-1]:
            lens.append(next_len)
        current = next_len

    if lens[-1] != min_seq_len:
        lens.append(min_seq_len)
    return lens


def batch_sizes_by_power(min_batch_exp: int, max_batch_exp: int) -> list[int]:
    if min_batch_exp < 0:
        raise ValueError("--min-batch-exp must be non-negative.")
    if max_batch_exp < min_batch_exp:
        raise ValueError("--max-batch-exp must be at least --min-batch-exp.")
    return [2**exp for exp in range(min_batch_exp, max_batch_exp + 1)]


def make_shapes(
    batch_sizes: list[int],
    seq_lens: list[int],
    min_elements: int,
    max_elements: int,
) -> list[Shape]:
    shapes = [
        Shape(batch_size=batch_size, seq_len=seq_len)
        for batch_size in batch_sizes
        for seq_len in seq_lens
        if min_elements <= batch_size * seq_len < max_elements
    ]
    if not shapes:
        raise ValueError("No legal shapes remain after applying the B * L limits.")
    return shapes


def shuffled_schedule(shapes: list[Shape], iterations: int, rng: random.Random) -> list[Shape]:
    if iterations < 0:
        raise ValueError("iterations must be non-negative.")
    schedule = [shape for _ in range(iterations) for shape in shapes]
    rng.shuffle(schedule)
    return schedule


def block_until_ready(value) -> None:
    if isinstance(value, (tuple, list)):
        for item in value:
            block_until_ready(item)
    elif hasattr(value, "block_until_ready"):
        value.block_until_ready()


def sync_torch(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def torch_to_jax(tensor: torch.Tensor) -> jax.Array:
    tensor = tensor.detach()
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    try:
        return jax.dlpack.from_dlpack(tensor, copy=False)
    except ValueError as exc:
        if "requires a copy" not in str(exc):
            raise
        return jax.dlpack.from_dlpack(tensor, copy=None)


def jax_to_torch(array: jax.Array, device: torch.device) -> torch.Tensor:
    tensor = torch.from_dlpack(array, copy=False)
    if tensor.device.type != device.type:
        tensor = tensor.to(device)
    return tensor


def make_inputs(
    shape: Shape,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    parameters_bp = torch.tensor(
        FSRS7_DEFAULT_35_VALUES,
        dtype=torch.float32,
        device=device,
    ).expand(shape.batch_size, -1).contiguous()
    parameters_bp.requires_grad_(True)

    elapsed_bl = torch.rand(
        (shape.batch_size, shape.seq_len),
        generator=generator,
        dtype=torch.float32,
    )
    elapsed_bl = (elapsed_bl * 0.25 + 0.01).to(device)
    elapsed_bl[:, 0] = 0.0

    rating_bl = torch.randint(
        1,
        5,
        (shape.batch_size, shape.seq_len),
        generator=generator,
        dtype=torch.int32,
    )
    rating_bl = rating_bl.to(device)
    rating_bl[:, 0] = 4

    seq_lens = torch.full((shape.batch_size,), shape.seq_len, dtype=torch.int32, device=device)
    mask_b = torch.ones((shape.batch_size,), dtype=torch.float32, device=device)
    epoch_lens_b = torch.ones((shape.batch_size,), dtype=torch.float32, device=device)

    inputs = (parameters_bp, elapsed_bl, rating_bl, seq_lens, mask_b, epoch_lens_b)
    sync_torch(device)
    return inputs


def loss_and_prediction_and_grad_from_torch(
    inputs: tuple[torch.Tensor, ...],
    device: torch.device,
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    jax_inputs = tuple(torch_to_jax(tensor) for tensor in inputs)
    (loss, prediction), grad = fsrs_v7_jax.loss_and_prediction_and_grad(*jax_inputs)
    block_until_ready((loss, prediction, grad))
    return (
        (jax_to_torch(loss, device), jax_to_torch(prediction, device)),
        jax_to_torch(grad, device),
    )


def time_call(call, device: torch.device) -> tuple[float, object]:
    start = time.perf_counter()
    value = call()
    sync_torch(device)
    return (time.perf_counter() - start) * 1000.0, value


def validate_result(shape: Shape, value: object) -> None:
    (loss_value, _prediction), grad = value
    if not bool(torch.isfinite(loss_value).all().item()):
        raise RuntimeError(f"Non-finite loss for B={shape.batch_size}, L={shape.seq_len}.")
    if not bool(torch.isfinite(grad).all().item()):
        raise RuntimeError(f"Non-finite gradient for B={shape.batch_size}, L={shape.seq_len}.")


def run_shape_call(
    inputs: tuple[torch.Tensor, ...],
    device: torch.device,
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    return loss_and_prediction_and_grad_from_torch(inputs, device)


def warmup_shapes(
    schedule: list[Shape],
    device: torch.device,
    seed: int,
) -> None:
    checked_shapes: set[Shape] = set()
    progress = tqdm(
        enumerate(schedule),
        total=len(schedule),
        desc="Warming shapes",
        unit="call",
        smoothing=0.03,
        file=sys.stdout,
    )
    for index, shape in progress:
        progress.set_postfix_str(f"B={shape.batch_size} L={shape.seq_len}")
        inputs = make_inputs(shape, device, seed + index)
        value = run_shape_call(inputs, device)
        if shape not in checked_shapes:
            validate_result(shape, value)
            checked_shapes.add(shape)


def benchmark_shapes(
    schedule: list[Shape],
    device: torch.device,
    seed: int,
) -> dict[Shape, list[float]]:
    samples_by_shape: dict[Shape, list[float]] = {}
    checked_shapes: set[Shape] = set()
    progress = tqdm(
        enumerate(schedule),
        total=len(schedule),
        desc="Benchmarking shapes",
        unit="call",
        smoothing=0.03,
        file=sys.stdout,
    )
    for index, shape in progress:
        progress.set_postfix_str(f"B={shape.batch_size} L={shape.seq_len}")
        inputs = make_inputs(shape, device, seed + index)
        elapsed_ms, value = time_call(
            lambda: run_shape_call(inputs, device),
            device,
        )
        if shape not in checked_shapes:
            validate_result(shape, value)
            checked_shapes.add(shape)
        samples_by_shape.setdefault(shape, []).append(elapsed_ms)
    return samples_by_shape


def summarize(samples_by_shape: dict[Shape, list[float]]) -> list[ShapeResult]:
    results = []
    for shape, samples in samples_by_shape.items():
        timing = Timing.from_samples(samples)
        results.append(
            ShapeResult(
                shape=shape,
                samples=len(samples),
                timing=timing,
                rows_per_second=shape.batch_size / (timing.mean_ms / 1000.0),
                elements_per_second=shape.elements / (timing.mean_ms / 1000.0),
                us_per_row=timing.mean_ms * 1000.0 / shape.batch_size,
            ),
        )
    return sorted(results, key=lambda result: (result.shape.batch_size, result.shape.seq_len))


def write_results(
    path: Path,
    results: list[ShapeResult],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    jax_device = str(jax.devices()[0])
    rows = [
        {
            "B": result.shape.batch_size,
            "L": result.shape.seq_len,
            "mean_ms": result.timing.mean_ms,
            "jax_device": jax_device,
        }
        for result in results
    ]
    path.write_text(json.dumps(rows, indent=2) + "\n")


def print_summary(results: list[ShapeResult], output: Path) -> None:
    print(f"wrote {len(results)} shape timings to {output}")
    header = (
        "B".rjust(8),
        "L".rjust(5),
        "mean_ms".rjust(10),
        "median_ms".rjust(10),
        "rows/s".rjust(12),
        "M elems/s".rjust(12),
    )
    print(" ".join(header))
    for result in results:
        row = (
            f"{result.shape.batch_size:8d}",
            f"{result.shape.seq_len:5d}",
            f"{result.timing.mean_ms:10.3f}",
            f"{result.timing.median_ms:10.3f}",
            f"{result.rows_per_second:12.0f}",
            f"{result.elements_per_second / 1_000_000.0:12.3f}",
        )
        print(" ".join(row))


def main() -> None:
    args = parse_args()
    if args.warmup < 1:
        raise ValueError("--warmup must be at least 1 so every shape hits JIT before timing.")
    if args.repeat < 1:
        raise ValueError("--repeat must be at least 1.")

    batch_sizes = batch_sizes_by_power(args.min_batch_exp, args.max_batch_exp)
    seq_lens = seq_lens_by_ratio(args.max_seq_len, args.min_seq_len, args.seq_len_ratio)
    min_elements = 2**args.min_elements_exp
    max_elements = 2**args.max_elements_exp
    shapes = make_shapes(batch_sizes, seq_lens, min_elements, max_elements)

    rng = random.Random(args.seed)
    warmup_schedule = shuffled_schedule(shapes, args.warmup, rng)
    benchmark_schedule = shuffled_schedule(shapes, args.repeat, rng)

    device = choose_torch_device(args.device)
    output = args.output if args.output is not None else default_output_path()
    print(
        f"shapes={len(shapes)} warmup_calls={len(warmup_schedule)} "
        f"timed_calls={len(benchmark_schedule)} jax_backend={jax.default_backend()} "
        f"torch_device={device}"
    )
    print(f"seq_lens={seq_lens}")
    print(f"batch_sizes={batch_sizes}")
    print(f"excluded shapes where B * L < {min_elements} or B * L >= {max_elements}")
    print(
        "timed function: torch inputs -> jax loss_and_prediction_and_grad "
        "-> torch outputs, input creation outside timing"
    )

    warmup_shapes(warmup_schedule, device, seed=args.seed + 1_000_000)
    samples_by_shape = benchmark_shapes(benchmark_schedule, device, seed=args.seed + 2_000_000)
    results = summarize(samples_by_shape)
    write_results(output, results)
    print_summary(results, output)


if __name__ == "__main__":
    main()
