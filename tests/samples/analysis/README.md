# Test data samples

Used by analysis tests.

## Data files: traceback_xx/

### syn_params

Sample synapse parameters:
- DataFrame
- indices: (layer, pre, post)
- columns: [`w`]
- dtype: np.float_

### gradients

Sample excitatory firing rates (activation gradients):
- DataArray
- shape (`num_layers`, `num_nrns`)
- dtype: np.float_

### sensitivities

Expected sensitivities w.r.t. last layer (matched to `syn_params_xx`):
- NDArray
- dtype: np.float_
- shape (`num_inputs`,)

### config

Associated config used to generate test samples

Keys:
- `num_layers`
- `num_nrns`
- `num_inputs`
- `p_active`
- `p_connect`
- `seed`
- `post_ids`
