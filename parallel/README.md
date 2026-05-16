# Parallel Prepare LMDB Layout

`parallel/prepare.py` writes one packed tensor blob per user. The blob is saved
with `torch.save(dict[str, torch.Tensor], buffer)` under `{user_id}_packed`, so
read it back with `torch.load(BytesIO(raw), weights_only=True, map_location="cpu")`.

The database path is `parallel_db`; the map size is `50_000_000_000` bytes.

## Raw Review Tensors

These tensors come directly from `revlogs/user_id=<user_id>` before feature
engineering.

| Blob Entry | Dtype | Shape | Meaning |
| --- | --- | --- | --- |
| `ratings` | `torch.int8` | `[raw_reviews]` | Raw `rating` column. |
| `elapsed_days_int` | `torch.int32` | `[raw_reviews]` | Raw `elapsed_days` column. |
| `elapsed_days_real` | `torch.float32` | `[raw_reviews]` | `elapsed_seconds / 86400`, so `1.0 == 1 day`. |

## Test Tensors

These tensors are built from `create_features(df.copy(), config=config)`. The
copy is intentional: feature creation mutates the dataframe.

| Blob Entry | Dtype | Shape | Meaning |
| --- | --- | --- | --- |
| `test_index` | `torch.int32` | `[test_reviews_total]` | Concatenated test reviews for all TSCV splits, as `review_th - 1`. |
| `rmse_bins` | `torch.int8` | `[test_reviews_total]` | RMSE-bin category code for each entry in `test_index`. |
| `split` | `torch.int32` | `[n_splits + 1]` | 0-indexed test range starts for each split, plus `1_000_000_000` sentinel. |

`test_index` and `rmse_bins` have matching positions. For split `s`, the test
range starts at `split[s]` and ends before `split[s + 1]`.

## Training Batch Tensors

For each TSCV split, `prepare.py` reconstructs the same training layout used by
`script.py`'s `Trainer` and `fsrs_optimizer.BatchDataset`:

1. Start with `feature_df.iloc[train_index]`.
2. Apply `model.filter_training_data(train_set)`.
3. Drop rows whose sequence length is greater than `config.max_seq_len`.
4. Stable-sort by sequence length ascending, matching `BatchDataset`.
5. Chunk into batches with `getattr(model, "batch_size", config.batch_size)`.

Note: the installed `fsrs_optimizer.BatchDataset` sorts ascending by `seq_len`,
so batch `0` contains the shortest remaining sequences, not the longest.

| Blob Entry | Dtype | Shape | Meaning |
| --- | --- | --- | --- |
| `train_index` | `torch.int32` | `[train_reviews_total]` | Concatenated train review indices, as `review_th - 1`, in batch-layout order. |
| `train_batch_lengths` | `torch.int32` | `[train_batches_total]` | Length of each batch in `train_index`. |
| `train_split_lengths` | `torch.int32` | `[n_splits]` | Number of training batches for each TSCV split. |
| `batch_order` | `torch.int32` | `[100 * train_batches_total]` | Concatenated shuffled batch ids for 100 epochs per split. |
| `batch_order_epochs` | `torch.int32` scalar | `[]` | Number of epoch permutations stored, currently `100`. |

## Reconstructing Batches

For split `s`:

```python
batch_counts = train_split_lengths
batch_start = batch_counts[:s].sum()
batch_end = batch_start + batch_counts[s]

split_batch_lengths = train_batch_lengths[batch_start:batch_end]

review_start = train_batch_lengths[:batch_start].sum()
review_end = review_start + split_batch_lengths.sum()
split_train_index = train_index[review_start:review_end]
```

Batch `b` within split `s` is:

```python
local_start = split_batch_lengths[:b].sum()
local_end = local_start + split_batch_lengths[b]
batch_review_indices = split_train_index[local_start:local_end]
```

The shuffled order for epoch `e` in split `s` is:

```python
epochs = int(batch_order_epochs)
order_start = epochs * batch_counts[:s].sum() + e * batch_counts[s]
order_end = order_start + batch_counts[s]
epoch_batch_order = batch_order[order_start:order_end]
```

`epoch_batch_order` contains batch ids local to that split. Use each id with
`split_batch_lengths` and `split_train_index` as shown above.

## Completion Key

`{user_id}_packed` contains the tensor dictionary. `{user_id}_done` is written
as raw bytes (`b"true"`) after the packed blob is saved.

## Global Metadata

`metadata_user_max_train_split_lengths` is a database-level key, not a per-user
blob entry. It stores a `torch.int32` vector with one entry per prepared user,
in the same order as `user_ids` in `prepare.py`. Each entry is that user's
maximum value from `train_split_lengths`.
