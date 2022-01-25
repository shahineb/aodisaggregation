"""
Description : Aggregate together in dataframe outputs from multiple runs.

Usage: aggregate_results.py  [options] --i=<input_dir> --o=<output_dir>

Options:
  --i=<input_dir>                  Path to directory where grid search outputs are saved.
  --o=<output_dir>                 Output directory.
  --tex                            Outputs latex version of scores.
"""
import os
import yaml
import logging
from glob import glob
from docopt import docopt
import pandas as pd
from utils import flatten_dict


def main(args):
    # Get list of scores directories for each random seed
    scores_dirs = glob(args['--i'] + 'seed_*')

    # Initialize empty arrays for scores
    seeds = []
    scores = []

    # Iterate to load scores in
    for score_dir in scores_dirs:
        # Read seed value
        with open(os.path.join(score_dir, 'cfg.yaml'), "r") as f:
            cfg = yaml.safe_load(f)
            assert cfg['model']['seed'] == cfg['training']['seed']
            seeds.append(cfg['model']['seed'])

        # Load scores as flat dict
        with open(os.path.join(score_dir, 'scores.metrics'), "r") as f:
            score = flatten_dict(yaml.safe_load(f))
            scores.append(score)

    # Encapsulate in dataframe and aggregate
    scores_df = pd.DataFrame(data=scores, index=seeds).aggregate(['mean', 'std'])

    # Dump aggregate scores
    scores_df.to_json(os.path.join(args['--o'], 'scores.json'))
    if args['--tex']:
        scores_df.to_latex(os.path.join(args['--o'], 'scores.tex'))
    logging.info(f"Dumped aggregate scores under {args['--o']}")


if __name__ == "__main__":
    # Read input args
    args = docopt(__doc__)

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logging.info(f'Arguments: {args}\n')

    # Create output directory if doesn't exists
    os.makedirs(args['--o'], exist_ok=True)

    # Run session
    main(args)
