import yaml
from config_valid import Config

def get_diz_trials(nome_file_config="configurations/Final1.yml"):
    num_nodi = 10
    c = Config(nome_file_config)
    # c.conf['graph_dataset']['Num_nodes'] = [num_nodi]
    #c.conf['graph_dataset']['list_exponents'] = [-2.2, -2.9]
    c.conf['model']['autoencoder'] = False
    c.conf['model']['autoencoder_confmodel'] = True   # quando inserisco confmodel serve anche RELU
    c.conf['model']['autoencoder_graph_ae'] = False
    diz_trials = {'graph_dataset.ERmodel': [True],
                  'graph_dataset.confmodel': [False],
                  'graph_dataset.sbm': [False],
                  'graph_dataset.regular': [False],
                  'graph_dataset.real_dataset': [False],
                  'graph_dataset.Num_nodes': [[num_nodi] * 4, [num_nodi] * 7, [[num_nodi, int(num_nodi / 2)]] * 3],  # per lo SBM: num nodi * num classi * num comunità
                  'model.GCNneurons_per_layer': [  # [1, 32, 16, len(c.conf['graph_dataset']['list_exponents'])],
                      # [1, 32, 16, len(c.conf['graph_dataset']['list_p'])],
                      # [1, 32, 16, len(c.conf['graph_dataset']['community_probs'])],
                      #[1, 16, 16, 16, 16, 16],
                      [1, 64, 64, 64, 32],
                      #[1, 64, 64, 32]
                      #[1, 64, 64, 64, ],
                      #[1, 16, 16, 16, 16],
                      #[1, 16, 16, 16]
                      # [1, 256, 256, 128, 64, 16],
                      # [1, 64, 32, 16, 16],
                      # [1, 64, 32, 16, 2],
                      # [1, 32, 16, 8, 8],
                      # [1, 32, 16, 8, 2],
                      # [1, 16, 8, 4, 4],
                      # [1, 8, 8, 4, 2],
                  ],
                  # 'model.init_weights': ['xavier_normal', 'eye'],
                  # 'model.freezeGCNlayers': [False],
                  # 'model.last_layer_dense': [False],
                  #'training.learning_rate': [0.0001, 0.0005],
                  #'training.optimizer': ['SGD', 'ADAM']
                  }
    return c, diz_trials

def modify_some_trials(diz_trials, **kwargs):
    for k, v in kwargs.items():
        diz_trials[k] = v
    return diz_trials

def load_trials_edits(yml_file):
    edits = yaml.safe_load(open(yml_file))
    return edits
