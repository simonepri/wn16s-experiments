ID_TO_REL = [
  'hypernyms', 'instance_hypernyms',
  'hyponyms', 'instance_hyponyms',
  'member_holonyms', 'substance_holonyms', 'part_holonyms',
  'member_meronyms', 'substance_meronyms', 'part_meronyms',
  'attributes',
  'entailments',
  'causes',
  'also_sees',
  'verb_groups',
  'similar_tos'
]

relations = [{
  'name': rel_name, 'lhs': 'all', 'rhs': 'all',
  'sym_operator': 'projection',
  'operator': 'projtrans',
} for rel_name in ID_TO_REL]

def get_torchbiggraph_config():
  config = dict(
    # I/O data
    entity_path = 'model/wn16s',
    edge_paths = [],
    checkpoint_path = 'model/wn16s',
    init_path = None,

    # Graph structure
    entities = {
      'all': {'num_partitions': 1},
    },
    relations = relations,

    # Scoring model
    dimension = 128,
    max_norm = 1.0,
    global_emb = False,
    comparator = 'l2',
    bias = False,

    # Training
    num_epochs = 15,
    batch_size = 1000,
    num_batch_negs = 150,
    num_uniform_negs = 0,
    loss_fn = 'ranking',
    margin = 0.1,
    lr = 0.1,

    # Evaluation during training
    eval_fraction = 0,
  )

  return config
