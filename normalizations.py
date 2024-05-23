from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from scipy.stats import rankdata
from graph_generation import rndm



# Definizione della funzione di normalizzazione
def normalize_sequence(sequence, sottrai_minimo=True):
    max_value = max(sequence)
    min_value = min(sequence)
    scala = (max_value - min_value)
    #print(f"scala {scala}")
    normalized_sequence = (sequence - min_value) / scala
    if not sottrai_minimo:
        normalized_sequence = (sequence) / (max_value - min_value)
    #normalized_sequence = [(x - min_value) / (max_value - min_value) for x in sequence]
    return normalized_sequence, scala


def cdf_normalization(degrees):
    # Utilizza rankdata per ottenere i ranghi dei gradi (la loro posizione nella distribuzione cumulativa)
    ranks = rankdata(degrees, method='average')
    #ranks = np.argsort(np.argsort(degrees))
    # Calcola la CDF per ciascun grado
    cdf_values = (ranks - 1) / len(degrees)
    return cdf_values

def power_transform(sequence, exponent, **kwargs):
    # Applica una trasformazione di potenza ai gradi per ridimensionare la gamma di valori
    transformed_degrees = np.power(sequence, exponent)
    # Normalizza i gradi trasformati su una scala da 0 a 1
    normalized_transformed_degrees, scala = normalize_sequence(transformed_degrees, **kwargs)
    return normalized_transformed_degrees, scala

def log_transform(sequence, **kwargs):
    transformed = np.log(sequence)
    normalized_transformed_degrees, scala = normalize_sequence(transformed, **kwargs)
    return normalized_transformed_degrees, scala

def probability_transform(sequence):
    degree_count = Counter(sequence)
    total_count = sum(degree_count.values())
    degree_prob_dict = {degree: count / total_count for degree, count in degree_count.items()}
    probs = np.array([degree_prob_dict[degree] for degree in sequence])
    return probs

def degree_normalizations(degrees):
    rank_norm = cdf_normalization(degrees)
    scale_norm, scala_linear = normalize_sequence(degrees)
    log_norm, scala_log = log_transform(degrees)
    probs_transf = probability_transform(degrees)
    #pred_probs = probability_transform(pred_degrees)
    # ci metto pure la nortmalizzazione rispetto al numero totale di link del grafo?
    return rank_norm, scale_norm, log_norm, probs_transf


############# gestione delle barre d'errore
def barre_errore(diffs, norm_rank):
    unique_ranks, integer_values = np.unique(norm_rank, return_inverse=True)
    # unique_ranks sono i valori unici del norm_rank,
    # La variabile integer_values è un array di interi dove ciascun intero rappresenta
    # l'indice del valore float corrispondente nell'array unique_ranks,
    # così come trovati nel norm_rank. gli interi si ripetono perché mappano norm_rank
    # ai suoi valori unici

    # Crea un dizionario per raggruppare gli elementi di diffs basandoti su 'unique_ranks'
    grouped_differences = defaultdict(list)
    for idx, group in enumerate(integer_values):
        float_key = unique_ranks[group]  # potevo scrivere anche == norm_rank[idx] ?
        grouped_differences[float_key].append(diffs[idx])
        # ora ho tutte le differenze (errori) per ogni valore unico del rank norm
        # cioè un valore di errore per ogni nodo che ha quello stesso grado

    # quì eseguo la media sui nodi che hanno lo stesso grado
    errori_mean = {k: np.mean(v) for k, v in grouped_differences.items()}
    errori_abs = {k: np.abs(v).mean() for k, v in grouped_differences.items()}
    dev_std = {k: np.std(v) for k, v in grouped_differences.items()}
    unique_mean_errors = [errori_mean[f] for f in unique_ranks]  # errori ordinati come unique_ranks
    unique_abs_mean_errors = [errori_abs[f] for f in unique_ranks]  # errori ordinati come unique_ranks
    unique_dev_std = [dev_std[f] for f in unique_ranks]  # deviazioni std ordinate

    return np.array(unique_mean_errors), unique_ranks, integer_values, unique_abs_mean_errors, unique_dev_std


def get_unique_ys_4_unique_errors(degrees, unique_floats, integer_values):
    unique_ys = np.array([degrees[integer_values == i][0] for i in range(len(unique_floats))])
    return unique_ys

def get_summed_ys_at_unique_probs(degrees, unique_ranks_avg, integer_values):
    summed_ys = np.array([degrees[integer_values == i].sum() for i in range(len(unique_ranks_avg))])
    return summed_ys

def get_error_bars_2plot(pred_deg, orig_deg, rank_norm, scale_norm, log_norm, probs_transf):
    diffs = pred_deg - orig_deg
    unique_mean_errors, unique_floats, integer_values, unique_sum_errors, unique_dev_std = barre_errore(diffs, rank_norm)
    unique_degrees = get_unique_ys_4_unique_errors(orig_deg, unique_floats, integer_values)
    unique_norm = get_unique_ys_4_unique_errors(scale_norm, unique_floats, integer_values)
    unique_logs = get_unique_ys_4_unique_errors(log_norm, unique_floats, integer_values)
    unique_prob = get_unique_ys_4_unique_errors(probs_transf, unique_floats, integer_values)
    return unique_mean_errors, unique_floats, unique_degrees, unique_norm, unique_logs, unique_prob







########################################à#        plot funciotns   ################àà

def plot_1(xrank, orig_deg, pred_deg, unique_xrank, unique_ydeg, unique_diffs, ax, plot_diff_bars=True, **kwargs):
    linestyle=''
    marker='.'
    markersize=3
    alpha=0.9
    pER = kwargs.get('pER')
    exp = kwargs.get('exp')
    if pER is not None:
        label = f"p: {pER}"
    elif exp is not None:
        label = f"exp: {exp}"
    else:
        label = ''
    #print(label)

    ax.plot(xrank, orig_deg, linestyle=linestyle, marker=marker, markersize=markersize, alpha=alpha, label=label)
    alpha = 0.4
    #ax.plot(xrank, pred_deg, linestyle=linestyle, marker=marker, markersize=markersize, alpha=alpha)
    if plot_diff_bars:
        ax.errorbar(unique_xrank, unique_ydeg, yerr=unique_diffs, linestyle=linestyle, marker=marker, markersize=markersize, alpha=alpha)#, label=label)

    plt.legend()
