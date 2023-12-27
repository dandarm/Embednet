from pathlib import Path
import os
import re
import csv
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from pandas import json_normalize
import yaml
import pickle
from collections import Counter

root_task = Path("/home/daniele/Documenti/Progetti/Networks/Embednet/head_train_tasks")
all_runs_csv = root_task / "head_run_tasks.csv"

def get_df_from_configs(df_input):
    list_dfs = []
    for index, row in df_input.iterrows():
        #leggo un config per volta: ho un df da una riga, che poi concateno in un unico df
        root = Path(row.values[0])
        conf = yaml.safe_load(open(root / "config.yml"))
        df = json_normalize(conf)
        df = df.astype('object')
        df.index = [index]
        list_dfs.append(df)
    final_df = pd.concat(list_dfs, axis=0)
    #filtered_df = final_df.dropna(subset=[('risultati', 'train_loss')])

    # risolvo quì le issue per i campi lista che vanno trasformati in tuple
    final_df['model.GCNneurons_per_layer'] = final_df['model.GCNneurons_per_layer'].apply(lambda x: tuple(x))
    final_df['graph_dataset.list_p'] = final_df['graph_dataset.list_p'].apply(lambda x: tuple(x))
    final_df['graph_dataset.Num_nodes'] = final_df['graph_dataset.Num_nodes'].apply(lambda x: tuple(x))

    return final_df



#####   Studio SEQUENZA DI GRADO

root_file = '_degseq_tot_epoch'

# select relevant files
def estrai_numero(nome_file):
    match = re.search(r'(\d+)', nome_file)
    return int(match.group()) if match else None


def get_seq_4_epochs(folder):
    lista_file = [file for file in os.listdir(folder) if file.startswith(root_file)]

    # Ordina i file in base al numero nel nome
    lista_file_ordinati = sorted(lista_file, key=estrai_numero)

    contenuti = []
    numeri_file = []
    for nome_file in lista_file_ordinati:
        array = np.load(os.path.join(folder, nome_file))
        contenuti.append(array)
        numero = estrai_numero(nome_file)
        numeri_file.append(numero)

        # with open(os.path.join(folder, nome_file), 'r') as file:
        #    contenuti.append(file.read())

    # estrai la seq di grado originale del dataset di input
    input_seq = np.load(os.path.join(folder, "_degseq_totale.npy")).ravel()
    
    # estrai anche altre informazioni dal fie di configurazione
    prop = get_other_info_from_config(folder)

    return contenuti, numeri_file, input_seq, prop

def get_other_info_from_config(folder):
    """ restituisce delle proprietà dal config come:
    il numero dei grafi
    """
    config_file_path = Path(folder) / "config.yml"
    conf = yaml.safe_load(open(config_file_path))
    num_grafi = conf['graph_dataset']['Num_grafi_per_tipo']
    prop = (num_grafi,)
    return prop
    


def calc_media_scarti(input_seq, diff_seq):
    """
    Calcola la media sui nodi aventi lo stesso grado, oltre che la deviazione standard
    valore_int è il grado (input) che viene usato come chiave del' dizionario per accumulare
    i valori del grado predetto
    """
    accumulo = {}
    for i, valore_int in enumerate(input_seq):
        float_val = diff_seq[i]

        if valore_int in accumulo:
            prev_somma = accumulo[valore_int]['somma']
            prev_conteggio = accumulo[valore_int]['conteggio']
            nuova_media = (prev_somma + float_val) / (prev_conteggio + 1)
            diff = float_val - nuova_media
            accumulo[valore_int]['somma'] += float_val
            accumulo[valore_int]['conteggio'] += 1
            accumulo[valore_int]['somma_quad'] += diff ** 2

        else:
            accumulo[valore_int] = {'somma': float_val, 'conteggio': 1, 'somma_quad': 0}

    statistiche = {}
    for k, v in accumulo.items():
        media = v['somma'] / v['conteggio']
        varianza = v['somma_quad'] / v['conteggio']
        dev_std = np.sqrt(varianza)
        statistiche[k] = {'media': media, 'dev_std': dev_std}

    return statistiche


def plot_deg_seq(input_seq, pred_seq, ax=None):
    if ax is None:
        ax = plt.gca()
    plt.scatter(input_seq, pred_seq, label="Predicted")
    plt.scatter(input_seq, input_seq, label="Input", color='red')
    # Mostra il plot se stiamo utilizzando pyplot
    if ax is plt.gca():
        plt.show()


def plot_scarto(epoch, pred_seq, input_seq, ax=None, folder_path=None, ylim=None, xlim=None):
    if ax is None:
        ax = plt.gca()
    # plot_deg_seq(input_seq.ravel(), pred_seq, ax[0])
    diff = pred_seq - input_seq

    # ax[0].scatter(input_seq.ravel(), diff)

    stats = calc_media_scarti(input_seq, diff)
    # Estrazione di valori x, y e errori
    x_vals = list(stats.keys())
    y_vals = [stats[k]['media'] for k in x_vals]
    errori = [stats[k]['dev_std'] for k in x_vals]
    # Aggiungi i punti e le barre d'errore al plot esistente
    ax.errorbar(x_vals, y_vals, yerr=errori, fmt='o', ecolor='red', capsize=0, linestyle='None', color='blue', alpha=0.5, )
    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_title(f"Epoca {epoch}")

    if ylim:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(-7, 7)

    if xlim:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(0, 40)

    if folder_path is not None:
        ax.set_title(Path(folder_path).stem)

    if ax is plt.gca():
        plt.show()


def plot_curves_vs_features(plot_data, feature='feat',
                            titolo="", x_title="",  titolo_laterale="",
                            ax=None, y_lim=None, x_lim=None, logx=False, colorbar=True):
    if ax is None:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()

    colors, cm = get_color(plot_data)

    plot_data.sort(key=lambda x: x[feature])

    # Creazione del grafico
    #plt.figure(figsize=(10, 6))
    for i, data in enumerate(plot_data):
        # plt.plot(data['x'], data['y'],  color=colors[i])  #label=data['label'],
        plot_res = ax.errorbar(data['x'], data['y'], yerr=data['errori'], label=str(data.get('label')), fmt='o', color=colors[i],
                     ecolor='black', capsize=0, linestyle='None', alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=1)  # linea orizzontale a 0 che è il desiderato

    node_list = [d[feature] for d in plot_data]

    # Aggiungere una colorbar
    if colorbar:
        min_feat, max_feat = min(node_list), max(node_list)
        sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=min_feat, vmax=max_feat))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.25)
        plt.colorbar(sm, label=titolo_laterale, cax=cax)#, ax=ax)  # quì va bene usare plt invece di ax

    ax.set_title(titolo)
    ax.set_xlabel(x_title)
    y_title = "Relative difference degree sequence"
    ax.set_ylabel(y_title)
    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    #ax.legend()  #
    if logx:        
        ax.set_xscale('log')
    # plt.savefig(titolo)
    if ax is plt.gca():
        plt.show()


def get_color(plot_data):
    cm = plt.get_cmap('plasma')
    num_colors = len(plot_data)
    colors = [cm(1. * i / num_colors) for i in range(num_colors)]
    return colors, cm


def get_data_points_degrees_relative_difference(path):
    # carico i valori dello scarto y rispetto al grado x (con gli errori)
    pred_seq_4_epochs, epochs, input_seq, altre_prop = get_seq_4_epochs(path)
    pred_seq = pred_seq_4_epochs[-1]  # prendo la sequenza di grado dell'ultima epoca
    diff = (pred_seq - input_seq) / input_seq  # calcolo la differenza relativa tra la seq grado predetta e quella di input
    stats = calc_media_scarti(input_seq, diff)
    x_vals = list(stats.keys())  # contiene i valori unici del grado, mentre input_seq sono tutti i gradi di tutti i nodi (anche ripetuti)
    y_vals = [stats[k]['media'] for k in x_vals]
    errori = [stats[k]['dev_std'] for k in x_vals]
    
    degree_count = Counter(input_seq)
    # Normalizzazione dei conteggi -> probabilità!
    total_count = sum(degree_count.values())
    deg_prob = {degree: count / total_count for degree, count in degree_count.items()}
    
    num_grafi = altre_prop[0]
    tot_links = sum(input_seq) / 2 / num_grafi

    #print(f"input_seq: {type(input_seq), len(input_seq)}")
    #print(f"x_vals: {type(x_vals), len(x_vals)}")
    #print(f"deg_prob: {deg_prob, len(deg_prob)}")

    return x_vals, y_vals, errori, deg_prob, tot_links


def get_plot_data(many_training_paths, many_training_configs, **kwargs):
    plot_data = []
    for index, row in many_training_paths.iterrows():
        path = row.values[0]

        x_vals, y_vals, errori, deg_prob_diz, tot_links = get_data_points_degrees_relative_difference(path)

        # queste variabili sono chiamate tramite i kwargs
        pER = many_training_configs.loc[index]['graph_dataset.list_p'][0]
        exp = many_training_configs.loc[index]['graph_dataset.list_exponents'][0]
        num_nodes = many_training_configs.loc[index]['graph_dataset.Num_nodes'][0]

        #calcolo il numero di link dei grafi
        if kwargs.get('normalize_x'):
            num_medio_link_grafo = num_nodes * (num_nodes - 1) / 2 * pER            
            x_vals = np.array(x_vals) /  tot_links   # num_medio_link_grafo
         
        # normalizzazione diversa: ricavo la probabilità corrispondente per ciascun grado
        if kwargs.get('probs'):
            probabilities = [deg_prob_diz[degree] for degree in x_vals]
            x_vals = probabilities
        
        

        feat = kwargs.get('feat')

        plot_data.append({'x': x_vals, 'feat': eval(feat), 'y': y_vals, 'errori': errori})

    return plot_data