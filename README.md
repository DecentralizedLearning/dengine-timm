# dEngine timm extension

The following repo aims at extending [dEngine](https://github.com/DecentralizedLearning/dEngine) with model architectures from [timm](https://github.com/huggingface/pytorch-image-models) (PyTorch Image Models).

> **Disclaimer:** This project is an independent extension for dEngine. It is **not associated with, endorsed by, or affiliated with timm, its maintainers, or Hugging Face in any way.** timm is used here as a third-party dependency, like any other.

It plugs into dEngine's usual config system (`BUILTINS`, `load_experiment_from_yamls`, `load_engine`, etc.) and adds:
- a thin `create_model` wrapper around timm's model factory, so any timm architecture can be dropped in as a dEngine `client.local_model`
- ready-to-use experiment scripts showing centralized and decentralized (federated) training with a timm model on CIFAR-10

## Installation

```
pip install https://github.com/DecentralizedLearning/dengine-timm
```

This will also pull in `timm` as a dependency.

## Usage

A fully working example is provided in `main.py`. The important steps:

1. Import `create_model` from the plugin alongside dEngine's usual imports:

   ```python
   from dengine_timm import create_model
   ```

2. Instantiate any timm model just like you would with timm directly — every architecture name and argument supported by `timm.create_model` works here too:

   ```python
   MNASNET = create_model('mnasnet_050', pretrained=False)
   ```

3. Compose your config file list by mixing dEngine core configs (scenario, graph, partitioning, dataset) as usual:

   ```python
   FEDERATED_CONFIGS = [
       BUILTINS.CORE.SCENARIOS.DECENTRALIZED_HOMOGENOUS,
       BUILTINS.CORE.GRAPH.STAR_51,
       BUILTINS.CORE.PARTITIONING.NONIID,
       BUILTINS.CORE.DATASETS.MNIST,
   ]
   ```

4. Load and run as usual with `load_experiment_from_yamls` / `load_engine`, overriding `client.local_model` with your timm model wherever you'd normally define one:

   ```python
   experiment = load_experiment_from_yamls(
       files=FEDERATED_CONFIGS,
       overrides=convert_to_nested_dict({
           "name": "cifar10,FedAvg",
           "client.target": DecAvgClient.__name__,
           # any other client/training_engine/etc. overrides
       }),
       experiments_directory_root=str(simulation_args.output_directory.absolute()),
       seed=simulation_args.seed,
   )
   ```

Because model construction happens in plain Python via `create_model(...)`, you can swap architectures by changing a single string (e.g. `'mnasnet_050'` → `'resnet18'`, `'efficientnet_b0'`, etc.) — no extra config files needed.

## Example: centralized vs. decentralized CIFAR-10

`main.py` shows both a centralized baseline and several decentralized topologies (BA graph, ER graph, star/FedAvg) trained on CIFAR-10 with the same timm-backed model, run either sequentially or in parallel across processes:

```bash
python main.py --concurrent-runs 4
```

Useful flags:
- `--concurrent-runs N` — run up to `N` configurations in parallel via `multiprocessing.Pool`
- `--experiment_name_filter REGEX` — only run experiments whose name matches the given regex (can be passed multiple times)
- `--debug` — skip actual training/testing, useful for quickly checking that all configs load and wire together correctly
