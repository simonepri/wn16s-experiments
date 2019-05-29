import argparse
import os
from itertools import chain

import attr
import pkg_resources

import torchbiggraph.converters.utils as utils
from torchbiggraph.config import parse_config
from torchbiggraph.converters.import_from_tsv import convert_input_data
from torchbiggraph.eval import do_eval
from torchbiggraph.train import train

DATA_DIR = 'data'
WN16S_URL = 'https://github.com/simonepri/WN16S/releases/latest/download/WN16S.tgz'
FILENAMES = {
  'train': 'edge2id_all.tsv',
  'test': 'edge2id_all.tsv',
}
DEFAULT_CONFIG = pkg_resources.resource_filename('configs', 'configs/wn16s_config.py')

def convert_path(fname):
  basename, _ = os.path.splitext(fname)
  out_dir = basename + '_partitioned'
  return out_dir


def main():
  parser = argparse.ArgumentParser(description='Example on FB15k')
  parser.add_argument('--config', default = DEFAULT_CONFIG,
                      help='Path to config file')
  parser.add_argument('-p', '--param', action='append', nargs='*')
  parser.add_argument('--data_dir', default='data',
                      help='where to save processed data')
  parser.add_argument('--no-filtered', dest='filtered', action='store_false',
                      help='Run unfiltered eval')
  args = parser.parse_args()

  if args.param is not None:
      overrides = chain.from_iterable(args.param)  # flatten
  else:
      overrides = None

  data_dir = args.data_dir
  fpath = utils.download_url(WN16S_URL, DATA_DIR)
  utils.extract_tar(fpath)

  edge_paths = [os.path.join(DATA_DIR, name) for name in FILENAMES.values()]
  convert_input_data(
    args.config,
    edge_paths,
    lhs_col = 0,
    rhs_col = 2,
    rel_col = 1,
  )

  config = parse_config(args.config, overrides)

  train_path = [convert_path(os.path.join(DATA_DIR, FILENAMES['train']))]
  train_config = attr.evolve(config, edge_paths = train_path)

  train(train_config)

  eval_path = [convert_path(os.path.join(DATA_DIR, FILENAMES['test']))]
  relations = [attr.evolve(r, all_negs = True) for r in config.relations]
  eval_config = attr.evolve(config, edge_paths = eval_path, relations = relations)

  do_eval(eval_config)


if __name__ == '__main__':
  main()
