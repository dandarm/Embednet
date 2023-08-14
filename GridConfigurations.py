import numpy as np
import pandas as pd
import yaml
import json
from collections import defaultdict
from ast import literal_eval
from pandas import json_normalize

from config_valid import Config


class GridConfigurations():
    def __init__(self, config_class, dict_of_variables, verbose=False):
        self.verbose = verbose
        self.dict_of_variables = dict_of_variables
        self.config_class = config_class

        self.configs = None
        self.config_dataframe = None
        #self.ordinal_configs = None  # per tenere traccia esatta del config con la riga del dataframe
        self.embedding_dataframe = None  # per mantenere anche i node-embedding

    def make_configs(self):
        # Costruisco il dataframe come cross product di tutte le trials
        conf_df = self.config_class.get_config_dataframe()
        # dataframe di partenza
        variable, trial_list = next(iter(self.dict_of_variables.items()))
        df_cum = self.df_singlevariable_trials(conf_df, trial_list, variable)
        # dataframe successivi
        successivi = list(self.dict_of_variables.items())[1:]
        if len(successivi) > 0:
            for variable, trial_list in successivi:
                df_tmp = pd.DataFrame(columns=[variable])
                df_tmp[variable] = trial_list
                # devo togliere dal precedente df la colonna di variabili che vengono cambiate dal successivo df
                df_cum.drop(variable, axis=1, inplace=True)
                self.list2tuples(df_tmp)
                #possiamo finalmente fare il cross product
                df_cum = pd.merge(df_cum, df_tmp, how='cross')


        # Riprendo la gerarchia del file yaml
        df_cum.columns = df_cum.columns.str.split('.', expand=True)

        # voglio togliere prima le righe relative ai trial in cui dataset è real dove ho degenerazione di parametri per i grafi sintetici
        # così non sono neanche inserite nei configs
        # colonne eccetto quelle relative a graph_dataset:
        other_cols = df_cum.columns[df_cum.columns.get_level_values(0) != 'graph_dataset']
        # filtro i duplicati secondo tutte le altre colonne (poiché contiene liste devo usare astype(str))
        # queste che rimangono sono le righe utile tra tutte quelle con dataset reale
        real_ds_rows = df_cum['graph_dataset']['real_dataset']
        col2sort = [('graph_dataset','ERmodel'),('graph_dataset','confmodel'), ('graph_dataset','sbm'), ('graph_dataset','regular')]
        #indici_rimanenti = df_cum[real_ds_rows].astype(str).drop_duplicates(subset=other_cols, keep='first').index
        # voglio mantenere la riga dove tutti i graphtype sintetici siano False:
        # devo ordinarli mettendoli al primo posto, poi il drop ducplicates tieni la prima riga
        indici_rimanenti = df_cum[real_ds_rows].sort_values(col2sort, ascending=True).astype(str).drop_duplicates(subset=other_cols, keep='first').index
        # filtro via i ripetuti su dataset real e tengo quelli con dataset sintetici (~real_ds_rows)
        df_cum = pd.concat([df_cum.loc[indici_rimanenti], df_cum[~real_ds_rows]])
        df_cum.reset_index(drop=True, inplace=True)

        self.configs = []
        errors = 0
        righe_da_togliere = []
        num_trials = df_cum.shape[0]
        for i in range(num_trials):
            row = df_cum.iloc[i:i+1]  # se droppo nel ciclo iloc dà errore perché conta le righe vere, non l'indice: devo droppare dopo
            stacked = pd.DataFrame(row.stack(level=[0, 1], dropna=True)).reset_index(0).drop('level_0', axis=1)
            # ottengo un dizionario pronto da salvare in stringa json
            result = self.nested_dict()
            for r in stacked.itertuples():
                result[r.Index[0]][r.Index[1]] = r._1
            config_is_back = yaml.safe_load(json.dumps(result))
            config_is_back['device'] = 'gpu'

            try:
                config_class = Config.fromdict(config_is_back)  # faccio la validazione della config
                self.configs.append(config_class)
            except AssertionError as e:
                #raise e
                errors += 1
                if self.verbose:
                    print(f"AssertionError:\n {repr(e)}, tolgo la riga {i}\n")
                righe_da_togliere.append(i)
            except Exception as e:
                raise e
                errors += 1
                print("Controllare errori non di assert")
                if self.verbose:
                    print(f"Errore generale:\n {repr(e)}, tolgo la riga {i}\n")
                righe_da_togliere.append(i)

        df_cum.drop(index=righe_da_togliere, inplace=True)  # ricordare che è meglio non cambiare la lunghezza dell'array dentro al ciclo for

        print(f"{errors} configurazioni saltate su {num_trials}, farò i seguenti {num_trials - errors} training: ")
        for c in self.configs:
            print(f'{c.unique_train_name}')
        # posso tornare a un indice sequenziale, anzi devo perché per riempire il df coi risultati non so quali sono gli indici di riga, ma riempio sequenzialmente ciclando sulle configs
        df_cum = df_cum.reset_index().drop(columns=['index'])
        self.config_dataframe = df_cum
        self.sistema_dataframe_fields()


    def sistema_dataframe_fields(self):
        self.config_dataframe[('risultati', 'test_loss')] = '_'
        self.config_dataframe[('risultati', 'test_accuracy')] = '_'
        self.config_dataframe[('risultati', 'correlation_allclasses')] = '_'
        self.config_dataframe[('risultati', 'tau_allclasses')] = '_'
        self.config_dataframe[('risultati', 'data')] = '_'

    def df_singlevariable_trials(self, conf_df, trial_list, multicolums):
        nrows = len(trial_list)
        repeated = pd.DataFrame(np.repeat(conf_df.values, nrows, axis=0), columns=conf_df.columns)
        repeated.loc[:, multicolums] = trial_list
        return repeated

    # per risolvere il problema delle colonne che hanno 'list' come tipi di dato
    def list2tuples(self, df):
        s = (df.applymap(type) == list).all()
        cols2convert = s[s].index.to_list()
        for col in cols2convert:
            df[col] = df[col].apply(tuple)

    # funzione per gestire il dataframe in formato dizionario e prepararlo al json
    def nested_dict(self):
        return defaultdict(self.nested_dict)


    # def run_grid_search(self, trainer, saved_initial_weights=None, train_with_same_dataset=False, train_with_same_initial_weightparameter=False):
    #     for i, c in enumerate(self.configs):
    #         print(f'Run {i + 1}/{len(self.configs)}')
    #         try:
    #             if not train_with_same_dataset:
    #                 trainer.init_dataset()
    #                 trainer.load_dataset(trainer.gg.dataset)
    #             if train_with_same_initial_weightparameter:
    #                 model = trainer.init_GCN(saved_initial_weights)
    #                 trainer.load_model(model)
    #             else:
    #                 model = trainer.init_GCN()
    #                 trainer.load_model(model)
    #
    #             trainer.launch_training()
    #             # serve per get_corrs_training
    #             experiments.t = trainer
    #             experiments.c = c
    #             avg_corr_classes, avg_tau_classes = get_corrs_training()
    #             avg_corr_classes = np.array(avg_corr_classes).T
    #
    #             self.config_dataframe.at[i, ('risultati', 'test_loss')] = trainer.test_loss_list
    #             self.config_dataframe.at[i, ('risultati', 'test_accuracy')] = trainer.accuracy_list
    #             self.config_dataframe.at[i, ('risultati', 'correlation_allclasses')] = avg_corr_classes.tolist()
    #             self.config_dataframe.at[i, ('risultati', 'tau_allclasses')] = avg_tau_classes.tolist()
    #             self.config_dataframe.loc[i, ('risultati', 'data')] = pd.Timestamp.now()
    #
    #             # gc.config_dataframe.iloc[i:i+1].to_csv(outfile, index=False, mode='a', header=not os.path.exists(outfile))
    #
    #         except Exception as e:
    #             print(repr(e))
    #             raise e


def open_df_results(file_csv):
    df_data = pd.read_csv(file_csv, header=[0, 1])
    df_data.drop(columns=['device', 'logging'], inplace=True)
    senza_loss = df_data['risultati', 'test_loss'] == '_'
    print(f"Erano presenti {senza_loss.sum()} righe vuote")
    df_data.drop(df_data[senza_loss].index, inplace=True)

    # risolve il problema dovuto a celle che contengono liste, che in lettura vengono prese come stringhe
    df_data[('model', 'GCNneurons_per_layer')] = df_data[('model', 'GCNneurons_per_layer')].apply(literal_eval)
    df_data[('model', 'neurons_last_linear')] = df_data[('model', 'neurons_last_linear')].apply(literal_eval)
    df_data[('risultati', 'test_loss')] = df_data[('risultati', 'test_loss')].apply(literal_eval)
    df_data[('risultati', 'test_accuracy')] = df_data[('risultati', 'test_accuracy')].apply(literal_eval)
    #df_data[('risultati', 'correlation_allclasses')] = df_data[('risultati','correlation_allclasses')].apply(literal_eval)
    try:
        df_data[('risultati', 'correlation_class0')] = df_data[('risultati', 'correlation_class0')].apply(literal_eval)
        df_data[('risultati', 'correlation_class1')] = df_data[('risultati', 'correlation_class1')].apply(literal_eval)
    except:
        pass
        #print("Manca ")

    return df_data