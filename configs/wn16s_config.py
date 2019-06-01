def get_torchbiggraph_config():
  config = dict(
    # I/O data
    entity_path = 'data/wn16s',
    edge_paths = [],
    checkpoint_path = 'model/wn16s',
    init_path = None,

    # Graph structure
    entities = {
      'all': {'num_partitions': 1},
    },
    relations = [{
      'name': 'all_edges',
      'lhs': 'all',
      'rhs': 'all',
      'pre_operator': 'linear',
      'operator': 'translation',
    }],
    dynamic_relations = True,

    # Scoring model
    dimension = 128,
    max_norm = None,
    global_emb = False,
    comparator = 'cos',
    bias = False,

    # Training
    num_epochs = 2,
    batch_size = 1000,
    num_batch_negs = 100,
    num_uniform_negs = 0,
    loss_fn = 'ranking',
    margin = 0.1,
    lr = 0.1,

    # Evaluation during training
    eval_fraction = 0,
  )

  return config
