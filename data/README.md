# Connectome data

`raw/` is for source exports and is intentionally not versioned. `processed/` is for reproducible subgraphs.

The canonical neuron table requires `neuron_id`; supported optional columns are `cell_type`, `super_class`, `neurotransmitter`, `hemisphere`, `brain_region`, `is_sensory`, `is_motor`, `is_descending`, and `metadata`. The edge table requires `pre_neuron_id` and `post_neuron_id`; optional columns are `synapse_count`, `confidence`, `neurotransmitter`, and `region`.

CSV, TSV, parquet, and optional HDF5 tables are supported. Source-specific column names must be mapped explicitly through `load_connectome`; FlyDoom does not assume an undocumented export layout. Obtain data through official FlyWire or publication channels and comply with dataset terms.
