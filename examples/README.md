# Examples

## `custom_stream.py`

The smallest end-to-end example of plugging your own features into MTDB.

```bash
# in-memory toy stream (8 features, two regimes)
python examples/custom_stream.py

# your own arrays (.npz with at least 'train' and 'eval' [N, T, F])
python examples/custom_stream.py --data data/my_stream.npz
```

Data format and helper functions are documented in
[`mtdb/tasks/array.py`](../mtdb/tasks/array.py). For a full-quality run remove the
small overrides at the bottom of the script and use `configs/standard.yaml`
(or your own config).
