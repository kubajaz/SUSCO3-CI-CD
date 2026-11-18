# Matrix benchmark

Four variants of the same matrix product `A @ B`. Each mode stands in for a different commit, from a clean `np.dot` to increasingly expensive code. Matrices are drawn after `np.random.seed(42)`, so the same size and seed always rebuild the same inputs.

The default matrix size is 1536, the seed is 42, and the mode runs 10 times. The script prints each run's duration and the average.

```bash
python3 matrix_benchmark.py --mode baseline
python3 matrix_benchmark.py --mode commit_reg_low
python3 matrix_benchmark.py --mode commit_reg_med
python3 matrix_benchmark.py --mode commit_reg_high
```

Extra flags: `--repeats`, `--size`, `--seed`.

The same flags work in Docker:

```bash
docker build -t matrix-benchmark .
docker run --rm matrix-benchmark --mode commit_reg_low
```

## Modes

The percentages are a rough size of the extra CPU time at the default matrix size. They depend on the BLAS library and the core count.

### `baseline`

Clean vectorised product: one `np.dot(A, B)` call. This is the reference, with no extra work.

### `commit_reg_low`

A small regression, about 5–10% more CPU time.

- an unnecessary copy of `A` before the multiply,
- a short loop that only scans every eighth row of that copy,
- a second multiply of a thin strip (about 4% of the rows); that strip result is discarded.

### `commit_reg_med`

A medium regression, about 15–20% more CPU time.

- the matrix is split into two horizontal blocks,
- each block is cast to `float64` with a copy, then multiplied on its own,
- the blocks are concatenated at the end,
- a strip of about 5% of the columns is multiplied again and that result is discarded.

### `commit_reg_high`

A large regression, about 30–50% more CPU time.

- the product is computed in nested loops: 3 row blocks by 2 blocks of the inner dimension,
- every tile gets a new buffer and is then copied once more,
- tile results are accumulated into the output matrix.
