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
    relations = [{
      'name': '0',
      'lhs': 'all',
      'rhs': 'all',
      'sym_operator': 'projection',
      'operator': 'projtrans',
    },{
      'name': '1',
      'lhs': 'all',
      'rhs': 'all',
      'sym_operator': 'projection',
      'operator': 'projtrans',
    },{
      'name': '2',
      'lhs': 'all',
      'rhs': 'all',
      'sym_operator': 'projection',
      'operator': 'projtrans',
    },{
      'name': '3',
      'lhs': 'all',
      'rhs': 'all',
      'sym_operator': 'projection',
      'operator': 'projtrans',
    },{
      'name': '4',
      'lhs': 'all',
      'rhs': 'all',
      'sym_operator': 'projection',
      'operator': 'projtrans',
    },{
      'name': '5',
      'lhs': 'all',
      'rhs': 'all',
      'sym_operator': 'projection',
      'operator': 'projtrans',
    },{
      'name': '6',
      'lhs': 'all',
      'rhs': 'all',
      'sym_operator': 'projection',
      'operator': 'projtrans',
    },{
      'name': '7',
      'lhs': 'all',
      'rhs': 'all',
      'sym_operator': 'projection',
      'operator': 'projtrans',
    },{
      'name': '8',
      'lhs': 'all',
      'rhs': 'all',
      'sym_operator': 'projection',
      'operator': 'projtrans',
    },{
      'name': '9',
      'lhs': 'all',
      'rhs': 'all',
      'sym_operator': 'projection',
      'operator': 'projtrans',
    },{
      'name': '10',
      'lhs': 'all',
      'rhs': 'all',
      'sym_operator': 'projection',
      'operator': 'projtrans',
    },{
      'name': '11',
      'lhs': 'all',
      'rhs': 'all',
      'sym_operator': 'projection',
      'operator': 'projtrans',
    },{
      'name': '12',
      'lhs': 'all',
      'rhs': 'all',
      'sym_operator': 'projection',
      'operator': 'projtrans',
    },{
      'name': '13',
      'lhs': 'all',
      'rhs': 'all',
      'sym_operator': 'projection',
      'operator': 'projtrans',
    },{
      'name': '14',
      'lhs': 'all',
      'rhs': 'all',
      'sym_operator': 'projection',
      'operator': 'projtrans',
    },{
      'name': '15',
      'lhs': 'all',
      'rhs': 'all',
      'sym_operator': 'projection',
      'operator': 'projtrans',
    }],

    # Scoring model
    dimension = 128,
    max_norm = None,
    global_emb = False,
    comparator = 'cos',
    bias = False,

    # Training
    num_epochs = 1000,
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
