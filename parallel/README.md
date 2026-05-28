Requirements: Linux, or Windows with WSL and Docker.

Prepare the dataset.

```sh
docker compose --progress quiet run --rm srs-benchmark python -m parallel.prepare --processes 10
```

Run training and evaluation.

```sh
docker compose --progress quiet run --rm srs-benchmark parallel/run_enzyme_torch_extension.sh
```

If you encounter issues on WSL, try increasing the memory limit with `.wslconfig`:
https://learn.microsoft.com/windows/wsl/wsl-config
