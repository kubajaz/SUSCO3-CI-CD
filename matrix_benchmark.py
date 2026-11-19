#!/usr/bin/env python3
"""Matrix-multiplication benchmark with four energy-efficiency variants.

Each mode computes ``A @ B`` and stands in for a different commit of the
same operation, from a clean vectorised product to increasingly wasteful
code:

- ``baseline``: one ``np.dot`` call.
- ``commit_reg_low``: an unnecessary copy plus a leftover strip of work.
  Aims for roughly 5-10% more CPU time.
- ``commit_reg_med``: float64 casts and the product taken in row slices.
  Aims for roughly 15-20% more CPU time.
- ``commit_reg_high``: nested tile loops that allocate a fresh buffer on
  every tile. Aims for roughly 30-50% more CPU time.

The percentages are approximate. They move with matrix size, the BLAS
library, and how many cores that library uses. The constants in this
module are chosen for the default square size.

Import into a measurement script (CodeCarbon, a Prometheus exporter, and
so on) and time ``run`` yourself::

    from matrix_benchmark import MatrixBenchmark

    bench = MatrixBenchmark(size=3072, seed=42)
    product = bench.run("commit_reg_low")
"""

from __future__ import annotations

import argparse
import socketserver
import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

DEFAULT_SIZE = 3072
DEFAULT_SEED = 42
DEFAULT_REPEATS = 1
DEFAULT_DAEMON_HOST = "0.0.0.0"
DEFAULT_DAEMON_PORT = 9000

# Tuned at DEFAULT_SIZE on a multi-core OpenBLAS build. Smaller problems make
# the Python and allocation overhead a larger share of the runtime.
_LOW_STRIP_FRACTION = 0.04
_MED_ROW_BLOCKS = 2
_MED_EXTRA_COLUMN_FRACTION = 0.05
_HIGH_ROW_BLOCKS = 3
_HIGH_INNER_BLOCKS = 2

MODES = (
    "baseline",
    "commit_reg_low",
    "commit_reg_med",
    "commit_reg_high",
)

_Implementation = Callable[[np.ndarray, np.ndarray], np.ndarray]


def generate_matrices(
    size: int = DEFAULT_SIZE,
    seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic pair of square ``float64`` matrices.

    Calls ``np.random.seed(seed)`` before drawing both matrices, so the
    same ``size`` and ``seed`` always produce the same inputs. That call
    also reseeds NumPy's global generator.
    """
    if size < 1:
        raise ValueError("size must be a positive integer")
    np.random.seed(seed)
    matrix_a = np.random.rand(size, size)
    matrix_b = np.random.rand(size, size)
    return matrix_a, matrix_b


def multiply(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    mode: str = "baseline",
) -> np.ndarray:
    """Multiply ``matrix_a`` by ``matrix_b`` with the implementation ``mode`` names."""
    try:
        implementation = _IMPLEMENTATIONS[mode]
    except KeyError:
        known = ", ".join(MODES)
        raise ValueError(f"unknown mode {mode!r}; expected one of: {known}") from None

    matrix_a = np.asarray(matrix_a)
    matrix_b = np.asarray(matrix_b)
    if matrix_a.ndim != 2 or matrix_b.ndim != 2:
        raise ValueError("both inputs must be 2-D arrays")
    if matrix_a.shape[1] != matrix_b.shape[0]:
        raise ValueError(
            f"incompatible shapes {matrix_a.shape} and {matrix_b.shape}"
        )
    return implementation(matrix_a, matrix_b)


class MatrixBenchmark:
    """Holds one deterministic matrix pair and runs a chosen commit mode.

    Build it once, then call :meth:`run` from an external energy or timing
    loop. Generating the inputs inside the timed region would mix allocation
    cost into the commit comparison.
    """

    def __init__(self, size: int = DEFAULT_SIZE, seed: int = DEFAULT_SEED) -> None:
        self.size = size
        self.seed = seed
        self.matrix_a, self.matrix_b = generate_matrices(size, seed)

    def run(self, mode: str = "baseline") -> np.ndarray:
        """Execute one multiplication in ``mode`` and return the product."""
        return multiply(self.matrix_a, self.matrix_b, mode)


@dataclass(frozen=True)
class BenchmarkResult:
    """Durations, in seconds, for one mode repeated on one matrix pair."""

    mode: str
    size: int
    seed: int
    durations_s: list[float]
    average_s: float
    checksum: float


def benchmark(
    mode: str = "baseline",
    repeats: int = DEFAULT_REPEATS,
    size: int = DEFAULT_SIZE,
    seed: int = DEFAULT_SEED,
) -> BenchmarkResult:
    """Run ``mode`` ``repeats`` times and return the durations in seconds.

    ``checksum`` comes from the last product, which keeps that multiply
    live until the timer has stopped.
    """
    if repeats < 1:
        raise ValueError("repeats must be a positive integer")

    bench = MatrixBenchmark(size=size, seed=seed)
    durations: list[float] = []
    checksum = 0.0
    for _ in range(repeats):
        start = time.perf_counter()
        product = bench.run(mode)
        durations.append(time.perf_counter() - start)
        checksum = float(product[0, 0] + product[-1, -1])

    return BenchmarkResult(
        mode=mode,
        size=size,
        seed=seed,
        durations_s=durations,
        average_s=sum(durations) / len(durations),
        checksum=checksum,
    )


def _bounds(length: int, parts: int) -> list[tuple[int, int]]:
    """Split ``0 .. length`` into ``parts`` non-empty half-open ranges."""
    parts = max(1, min(parts, length))
    edges = [index * length // parts for index in range(parts + 1)]
    return list(zip(edges, edges[1:]))


def _keep(array: np.ndarray) -> None:
    """Read one entry so a discarded result stays live for the timed run."""
    if not np.isfinite(array.reshape(-1)[0]):
        raise RuntimeError("non-finite intermediate result")


def _multiply_baseline(matrix_a: np.ndarray, matrix_b: np.ndarray) -> np.ndarray:
    """Clean vectorised product."""
    return np.dot(matrix_a, matrix_b)


def _multiply_reg_low(matrix_a: np.ndarray, matrix_b: np.ndarray) -> np.ndarray:
    """Copy an operand, scan a few entries, and recompute a thin row strip.

    The copy feeds the real product. The strip is a leftover recomputation of
    about 4% of the rows and is discarded.
    """
    working = np.copy(matrix_a)
    total = 0.0
    stride = max(1, working.shape[0] // 8)
    for row in range(0, working.shape[0], stride):
        total += float(working[row, 0])
    if total != total:
        raise RuntimeError("non-finite matrix entry")

    product = np.dot(working, matrix_b)
    strip = max(1, int(round(working.shape[0] * _LOW_STRIP_FRACTION)))
    _keep(np.dot(working[:strip], matrix_b))
    return product


def _multiply_reg_med(matrix_a: np.ndarray, matrix_b: np.ndarray) -> np.ndarray:
    """Cast row slices to float64, multiply them, and recompute a column strip.

    The column strip is leftover slicing: it repeats part of the product and
    the result is discarded.
    """
    parts: list[np.ndarray] = []
    for start, stop in _bounds(matrix_a.shape[0], _MED_ROW_BLOCKS):
        panel = np.array(matrix_a[start:stop], dtype=np.float64, copy=True)
        parts.append(np.dot(panel, matrix_b))
    product = np.concatenate(parts, axis=0)
    columns = max(1, int(round(matrix_b.shape[1] * _MED_EXTRA_COLUMN_FRACTION)))
    _keep(np.dot(matrix_a, matrix_b[:, :columns]))
    return product


def _multiply_reg_high(matrix_a: np.ndarray, matrix_b: np.ndarray) -> np.ndarray:
    """Blocked product: a Python loop nest with a fresh buffer on every tile.

    Tiles stay large enough for the extra cost to land around 30-50% at
    the default matrix size.
    """
    rows, inner = matrix_a.shape
    columns = matrix_b.shape[1]
    result = np.zeros((rows, columns), dtype=np.result_type(matrix_a, matrix_b))
    for row_start, row_stop in _bounds(rows, _HIGH_ROW_BLOCKS):
        for inner_start, inner_stop in _bounds(inner, _HIGH_INNER_BLOCKS):
            tile = np.array(
                matrix_a[row_start:row_stop, inner_start:inner_stop],
                copy=True,
            )
            # Redundant second allocation of the same tile.
            tile = np.array(tile, copy=True)
            right = np.array(matrix_b[inner_start:inner_stop, :], copy=True)
            result[row_start:row_stop] += np.dot(tile, right)
    return result


_IMPLEMENTATIONS: dict[str, _Implementation] = {
    "baseline": _multiply_baseline,
    "commit_reg_low": _multiply_reg_low,
    "commit_reg_med": _multiply_reg_med,
    "commit_reg_high": _multiply_reg_high,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Time matrix-multiplication commit modes or serve them over TCP."
    )
    parser.add_argument(
        "--mode",
        default="baseline",
        choices=MODES,
        help="implementation variant to run (default: baseline)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help=f"timed repetitions (default: {DEFAULT_REPEATS})",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=DEFAULT_SIZE,
        help=f"square matrix dimension (default: {DEFAULT_SIZE})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"NumPy seed passed to np.random.seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="run as a TCP benchmark daemon instead of printing once",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_DAEMON_HOST,
        help=f"daemon bind address (default: {DEFAULT_DAEMON_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_DAEMON_PORT,
        help=f"daemon TCP port (default: {DEFAULT_DAEMON_PORT})",
    )
    return parser


def format_result(outcome: BenchmarkResult) -> str:
    lines = [
        f"mode: {outcome.mode}",
        f"size: {outcome.size}",
        f"seed: {outcome.seed}",
        f"repeats: {len(outcome.durations_s)}",
    ]
    for index, elapsed in enumerate(outcome.durations_s, start=1):
        lines.append(f"run {index}/{len(outcome.durations_s)}: {elapsed:.6f} s")
    lines.append(f"average: {outcome.average_s:.6f} s")
    lines.append(f"checksum: {outcome.checksum:.6f}")
    return "\n".join(lines)


SOCKET_MODES = {
    "1": "baseline",
    "2": "commit_reg_low",
    "3": "commit_reg_med",
    "4": "commit_reg_high",
}


class BenchmarkRequestHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        request = self.request.recv(1024)
        command = request.decode("ascii", errors="replace").strip()
        mode = SOCKET_MODES.get(command)

        if mode is None:
            response = "invalid command: send 1, 2, 3, or 4\n"
        else:
            server = self.server
            if not isinstance(server, BenchmarkDaemon):
                response = "daemon configuration error\n"
            else:
                with server.benchmark_lock:
                    outcome = benchmark(
                        mode=mode,
                        repeats=server.repeats,
                        size=server.size,
                        seed=server.seed,
                    )
                response = format_result(outcome) + "\n"

        self.request.sendall(response.encode("utf-8"))


class BenchmarkDaemon(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        repeats: int,
        size: int,
        seed: int,
    ) -> None:
        super().__init__(server_address, BenchmarkRequestHandler)
        self.repeats = repeats
        self.size = size
        self.seed = seed
        self.benchmark_lock = threading.Lock()


def run_daemon(host: str, port: int, repeats: int, size: int, seed: int) -> None:
    if port < 1 or port > 65535:
        raise SystemExit("port must be between 1 and 65535")
    with BenchmarkDaemon((host, port), repeats, size, seed) as server:
        print(f"benchmark daemon listening on {host}:{port}", flush=True)
        print(
            "1=baseline 2=small-regression 3=medium-regression "
            "4=big-regression",
            flush=True,
        )
        server.serve_forever()


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.repeats < 1:
        raise SystemExit("repeats must be a positive integer")
    if args.size < 1:
        raise SystemExit("size must be a positive integer")

    if args.daemon:
        run_daemon(args.host, args.port, args.repeats, args.size, args.seed)
        return

    outcome = benchmark(
        mode=args.mode,
        repeats=args.repeats,
        size=args.size,
        seed=args.seed,
    )
    print(format_result(outcome))


if __name__ == "__main__":
    main()
