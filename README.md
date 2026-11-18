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

## TCP daemon

The Docker image runs the benchmark as a daemon by default. The listener is
bound to all pod/container interfaces on port `9000`.

```bash
docker run --rm -p 9000:9000 matrix-benchmark
printf '1\n' | nc 127.0.0.1 9000
```

Send only one of these commands in a TCP connection:

- `1`: baseline
- `2`: small regression (`commit_reg_low`)
- `3`: medium regression (`commit_reg_med`)
- `4`: big regression (`commit_reg_high`)

The benchmark settings can be passed to the daemon:

```bash
python3 matrix_benchmark.py --daemon --host 127.0.0.1 --port 9000 \
  --repeats 3 --size 512
```

## Kubernetes

Build and publish an image named where the cluster can pull it, update
`spec.template.spec.containers[].image` in `k8s/deployment.yaml`, then apply:

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

The service is a `LoadBalancer`. Get its external address and send a command:

```bash
kubectl get service matrix-benchmark
printf '1\n' | nc <EXTERNAL-IP> 9000
```

For a local cluster or to test before the external load balancer is allocated,
use port-forwarding:

```bash
kubectl port-forward service/matrix-benchmark 9000:9000
printf '1\n' | nc 127.0.0.1 9000
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
