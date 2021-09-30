"""
Description : Aggregate together in dataframe outputs from grid search run.

Usage: aggregate_hyperparams_search_results.py  [options] --i=<input_dir> --o=<output_dir>

Options:
  --i=<input_dir>                  Path directory where grid search outputs are saved.
  --o=<output_dir>                 Output directory.
"""
import os
import yaml
import logging
from docopt import docopt
import pandas as pd
