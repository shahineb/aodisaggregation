# Reconstructing Aerosols Vertical Profiles with Aggregate Output Learning


## Getting started

- Run from root directory
```bash
$ python run_svgp_vertical_profile.py --cfg=cfg/svgp.yaml --o=path/to/output/directory --plot
```


## Reproduce experiments

- SVGP
Run the bash script to repeat the experiment with multiple seeds
```bash
$ bash repro/repro_svgp.sh --device=cpu
```

Aggregate results across seeds to recover results from the paper
```bash
$ python aggregate_results.py --i=experiments/data/outputs/svgp --o=experiments/data/outputs/svgp
```


## Installation

Code implemented in Python 3.8.0

#### Setting up environment

Create and activate environment
```bash
$ pyenv virtualenv 3.8.0 venv
$ pyenv activate venv
$ (venv)
```

Install dependencies
```bash
$ (venv) pip install -r requirements.txt
```
