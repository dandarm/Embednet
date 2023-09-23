import yaml
from config_valid import Config

def get_diz_trials(nome_file_config="configurations/Final1.yml"):
    num_nodi = 50
    c = Config(nome_file_config)
    print("primo config base da estendere con i trials validato.")
    #c.conf['graph_dataset']['list_exponents'] = [-2.2, -2.9]
    diz_trials = {'model.autoencoder': [False],
                  'model.autoencoder_confmodel': [True],
                  'model.autoencoder_mlpdecoder': [False],
                  'model.autoencoder_fullMLP': [False],
                  'model.last_layer_activation': ['RELU'],  # 'Identity'],  # una esclude l-altra con AE e AE_CM
                  'graph_dataset.ERmodel': [True],
                  'graph_dataset.confmodel': [False],
                  'graph_dataset.sbm': [False],
                  'graph_dataset.const_degree_dist': [False],
                  'graph_dataset.real_dataset': [False],
                  'graph_dataset.list_p': [[0.1]],  # [[0.1],[0.3],[0.5],[0.7],[0.9]]  #[0.1,0.2,0.5,0.8,0.9]
                  'graph_dataset.Num_nodes': [[num_nodi]],  # , [num_nodi]*5, [[num_nodi, int(num_nodi / 2)]] * 3],  # per lo SBM: num nodi * num classi * num comunità
                  'graph_dataset.Num_grafi_per_tipo': [20],
                  'model.GCNneurons_per_layer': [
                      # [1, 32, 16, len(c.conf['graph_dataset']['list_exponents'])],
                      # [1, 32, 16, len(c.conf['graph_dataset']['list_p'])],
                      # [1, 32, 16, len(c.conf['graph_dataset']['community_probs'])],
                      #[1, 16, 16, 16, 16, 16],
                      #[1, 64, 64, 64, 32],
                      #[1, 32, 32, 32]
                      #[1, 256, 64],
                      #[1, 64, 64, 32],
                      [1, 64, 64],
                      #[1, 16, 4],

                      #[1, 256, 64],
                      #[1, 256, 8]
                      #[1, 32, 32, 8],
                      #[1, 32, 32, 6],
                      #[1, 32, 32, 4],
                      #[1, 32, 32, 2],
                      #[1,32,16]
                      # [1, 16, 16, 16, 16],
                      # [1, 16, 16, 16]
                      #[1,8,8,8,8]
                      # [1, 256, 256, 128, 64, 16],
                      # [1, 64, 32, 16, 16],
                      # [1, 64, 32, 16, 2],
                      # [1, 32, 16, 8, 8],
                      # [1, 32, 16, 8, 2],
                      # [1, 16, 8, 4, 4],
                      # [1, 8, 8, 4, 2],
                      #[1, 32, 2],
                      #[1,4, 2],
                      #[1,2,2],
                      #[1,2]
                  ],
                  'model.init_weights': ['xavier_normal'],  # , 'eye'],
                  #'model.put_batchnorm': [False],
                  'model.put_graphnorm': [True],
                  #'training.learning_rate': [0.0001, 0.0005],
                  #'training.optimizer': ['SGD', 'ADAM']
                  #'training.loss': ['BCELoss'],  #'MSELoss'
                  #'training.shuffle_dataset': [False,True]
                  }
    return c, diz_trials

def get_diz_trial4test(nome_file_config="configurations/Final1.yml"):
    num_nodi = 20
    #[5],[7],[9],[13],[20],[30],[50],[75],[100],[150],[200],
    c = Config(nome_file_config)
    print("primo config base fatto")

    diz_trials = {#'model.autoencoder': [False],
                  #'model.autoencoder_confmodel': [True],
                  #'model.autoencoder_mlpdecoder': [False],
                  #'model.last_layer_activation': ['RELU'],
                  'graph_dataset.ERmodel': [True],
                  'graph_dataset.confmodel': [False],
                  'graph_dataset.sbm': [False],
                  'graph_dataset.regular': [False],
                  'graph_dataset.real_dataset': [False],
                  'graph_dataset.list_p': [[0.1],[0.2],[0.3],[0.4],[0.5],[0.6],[0.7],[0.8],[0.9]],
                  'graph_dataset.Num_nodes': [[num_nodi], [[num_nodi, int(num_nodi / 2)]] * 3],  # per lo SBM: num nodi * num classi * num comunità
                  #'graph_dataset.Num_grafi_per_tipo': [10, 20, 50, 100], #, 150, 200, 300],
                  #'model.GCNneurons_per_layer': [
                  #    # [1, 32, 16, len(c.conf['graph_dataset']['list_exponents'])],
                  #    # [1, 32, 16, len(c.conf['graph_dataset']['list_p'])],
                  #    # [1, 32, 16, len(c.conf['graph_dataset']['community_probs'])],
                  #    [1, 32, 32, 16]
                  #],
                  # 'model.init_weights': ['xavier_normal', 'eye'],
                  # 'model.freezeGCNlayers': [False],
                  # 'model.last_layer_dense': [False],
                  #'training.learning_rate': [0.0001, 0.0005],
                  #'training.optimizer': ['SGD', 'ADAM']
                  #'training.weigths4unbalanced_dataset': [True, False]
                  }
    return c, diz_trials

def modify_some_trials(diz_trials, **kwargs):
    for k, v in kwargs.items():
        diz_trials[k] = v
    return diz_trials

def load_trials_edits(yml_file):
    edits = yaml.safe_load(open(yml_file))
    return edits
