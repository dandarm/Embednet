import copy
import pandas as pd
from config_valid import Config, TrainingMode

def make_config_cm(config, trials):
    configs = []
    # for lr, layer, num_nodes in trials:
    for list_exponents in trials:
        config['graph_dataset']['list_exponents'] = list_exponents
        copia = copy.deepcopy(config)
        confclass = Config.fromdict(copia)
        configs.append(confclass)
    return configs



def config2df(dati_c):
    # dati_c è una lista di Config
    dati = [d.conf for d in dati_c]
    df_data = pd.DataFrame(columns=['range_p', 'Num_nodes', 'Num_grafi_totali', 'learning_rate', 'batch_size', 'neurons_per_layer', 'correlations', 'error', 'test_loss', 'distance_of_means', 'batch_norm'])
    df_data['range_p'] = [d['graph_dataset']['range_p'] for d in dati]
    df_data['Num_nodes'] = [d['graph_dataset']['Num_nodes'] for d in dati]
    df_data['Num_grafi_totali'] = [d['graph_dataset']['Num_grafi_totali'] for d in dati]
    df_data['learning_rate'] = [d['training']['learning_rate'] for d in dati]
    df_data['batch_size'] = [d['training']['batch_size'] for d in dati]
    df_data['neurons_per_layer'] = [d['model']['neurons_per_layer'] for d in dati]
    df_data['correlations'] = [d.get('correlations') for d in dati]
    df_data['error'] = [d.get('error') for d in dati]
    df_data['test_loss'] = [[round(l, 10) for l in d.get('test_loss') or []] for d in dati ]  # or [] : per il caso in cui d.get restituisca None
    df_data['distance_of_means'] = [d.get('distance_of_means') for d in dati]  # caso della classification
    #df_data['run_num'] = [d.get('run_num') for d in dati]
    df_data['batch_norm'] = [d['model']['put_batchnorm'] for d in dati]
    df_data['num_epochs'] = [d['training']['epochs'] for d in dati]
    df_data['run_number'] = [d.get('run number') for d in dati]
    df_data['average_corr_emb_degree'] = [d.get('average_corr_emb_degree') for d in dati]
    return df_data