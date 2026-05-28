Requirements: Linux or Windows with WSL, Docker.

Prepare the dataset.

```sh
docker compose --progress quiet run --rm srs-benchmark python -m parallel.prepare --processes 10
```

Run training/evaluation.

```sh
docker compose --progress quiet run --rm srs-benchmark parallel/run_enzyme_torch_extension.sh
```
