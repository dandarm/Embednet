import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from plt_parameters import get_colors_to_cycle_rainbow8, get_colors_to_cycle_rainbowN, get_colors_to_cycle_sequential
from matplotlib.lines import Line2D
from config_valid import TrainingMode



# region plots
def plot_metrics(embeddings, num_emb_neurons, training_mode):
    embeddings.get_metrics(num_emb_neurons)
    if num_emb_neurons == 1:
        # embeddings_per_cluster solo per distribuzione discreta
        graph_emb_perclass = embeddings.get_all_graph_emb_per_class()
        labels = embeddings.get_all_scalar_labels_per_class()
        plot_dim1(graph_emb_perclass, want_kde=False, labels=labels)
        node_emb_perclass = embeddings.get_all_node_emb_per_class()
        plot_dim1(node_emb_perclass, want_kde=False, labels=labels)
        if training_mode == TrainingMode.mode3:
            plot_correlation_error(embeddings)
    else:
        embeddings.calc_distances()
        plot_dimN(embeddings, 300)

def plot_dim1(embeddings_per_class, bins=10, want_kde=True, density=True, nomefile=None, labels=None, title=None):
    #plt.figure(figsize=(18, 6))  # , dpi=60)
    # trova lo stesso binning
    new_bins = np.histogram(np.hstack(embeddings_per_class), bins=bins)[1]

    for i, emb in enumerate(embeddings_per_class):
        h, e = np.histogram(emb, bins=new_bins, density=density)
        x = np.linspace(e.min(), e.max())
        if labels:
            lab = labels[i]
            color = get_colors_to_cycle_sequential(len(embeddings_per_class))[i]
            plt.bar(e[:-1], h, width=np.diff(e), ec='k', align='edge', label=f"{lab}", color=color,alpha=0.7)
        else:
            plt.bar(e[:-1], h, width=np.diff(e), ec='k', align='edge', alpha=0.7)
        if want_kde:
            kde = stats.gaussian_kde(emb)
            plt.plot(x, kde.pdf(x), lw=5, label='KDE')

    #plt.xlabel('p', fontsize=18)
    plt.xticks(fontsize=18)
    if title:
        plt.title(title)
    plt.legend()
    if nomefile:
        plt.savefig(nomefile, dpi=100)
    plt.show()

def plot_correlation_error(embeddings):
    #plt.figure(figsize=(12, 6))  # , dpi=60)
    plt.scatter(embeddings.training_labels, embeddings.graph_embedding_array.flatten())  # , s=area, c=colors, alpha=0.5)
    # correlazione tra target e prediction
    #correlaz = np.corrcoef(embeddings.embeddings_array.flatten(), embeddings.embedding_labels)[0, 1]
    #error = np.sqrt(np.sum((embeddings.embeddings_array.flatten() - embeddings.embedding_labels) ** 2))
    if embeddings.graph_correlation_per_class:
        cs = [round(c, 5) for c in embeddings.graph_correlation_per_class]
    else:
        cs = round(embeddings.total_graph_correlation, 5)
    plt.title(f"Corr = {cs}   -   Error  {round(embeddings.regression_error, 5)}")
    plt.savefig("correl_error.png", dpi=72)
    plt.show()

def plot_dimN(embeddings, bins):
    #plt.figure(figsize=(14, 6))
    if len(embeddings.inter_dists) > 0 and embeddings.intra_dists is not None:
        plt.hist(embeddings.inter_dists, bins=bins, alpha=0.8)
        plt.hist(embeddings.intra_dists, bins=bins, alpha=0.8)
    else:
        all_distances = [e[0] for e in embeddings.distances]
        plt.hist(all_distances, bins=bins)
    plt.savefig("dist_vectors.png", dpi=72)
    plt.show()





def plot_graph_emb_1D(emb_by_class, config_c, str_filename=None, show=True, ax=None, close=False, sequential_colors=False):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    # plt.scatter(graph_embeddings.embeddings_array[:,0], graph_embeddings.embeddings_array[:,1], s=0.1, marker='.')
    exps = [] # config_c.conf['graph_dataset']['list_exponents']

    for i, emb_class in enumerate(emb_by_class):
        hist = []
        exps.append(emb_class[0].exponent)
        for emb_pergraph in emb_class:
            hist.append(emb_pergraph.graph_embedding)
    #for emb_pergraph in emb_perclass1:
    #    redhist.append(emb_pergraph.graph_embeddings_array)
        hist = np.array(hist).flatten()
        #redhist = np.array(redhist).flatten()
        if sequential_colors:
            color = get_colors_to_cycle_sequential(len(emb_by_class))[i]
            ax.hist(hist, bins=30, label=f"exp {exps[i]}", color=color);
        else:
            ax.hist(hist, bins=30, label=f"exp {exps[i]}")

        #ax.hist(redhist, bins=30, color='red');

    ax.set_title(f'Graph Embedding')
    ax.legend(loc=1, prop={'size': 6})  #custom_lines, [f"exp {e}" for e in exps])
    if str_filename:
        plt.savefig(str_filename)
    if show:
        plt.show()
    if close:
        plt.close()


def plot_graph_emb_3D(embeddings, config_c, filename=None, show=True, close=False, sequential_colors=False):
    fig = plt.figure(figsize=(20, 6))
    ax = fig.add_subplot(projection='3d')
    #num_nodi_totali = len(emb_by_class) * len(emb_by_class[0]) * len(emb_by_class[0][0].node_embedding_array)
    #alpha_value = min(1, 3000/num_nodi_totali)
    #labels = [] #config_c.conf['graph_dataset']['list_exponents']
    #custom_lines = []
    graph_emb_perclass = embeddings.get_all_graph_emb_per_class()
    # if graph_emb_perclass.ndim == 2:  # non è suddiviso per classe e quindi non plot color
    #     a, b, c = embedding.T
    #     ax.scatter(a, b, c)
    # graph_emb_perclass.ndim == 3:  # ho le classi
    for i, emb_class in enumerate(graph_emb_perclass):
        sc = []
        #labels.append(emb_class[0].graph_label)
        for graph_emb in emb_class:
            sc.append(graph_emb)
        sc = np.array(sc).flatten()
        a, b, c = sc.T
        ax.scatter(a,b,c, bins=30, label=f"{graph_emb.graph_label}")  # alpha=alpha_value,

        #color = get_colors_to_cycle_rainbow8()[i % 8]
        #if sequential_colors:
        #    color = get_colors_to_cycle_sequential(len(emb_by_class))[i]

    ax.set_title(f'Graph Embedding')
    ax.legend(loc=1)  #, [f"exp {e}" for e in exps])
    # plt.yscale('log')
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    if show:
        plt.show()
    if close:
        plt.close()


def plot_graph_emb_1D_continuousregression(embedding_class, config_c, str_filename=None, show=True, ax=None, close=False, sequential_colors=False):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    graph_embedding = embedding_class.graph_embedding.flatten()
    labels = embedding_class.training_labels
    ax.hist([graph_embedding, labels], bins=10)


        # if sequential_colors:
        #     color = get_colors_to_cycle_sequential(len(embs))[i]
        #     ax.hist(hist, bins=30, color=color);



    ax.set_title(f'Graph Embedding')
    ax.legend(loc=1, prop={'size': 6})  #custom_lines, [f"exp {e}" for e in exps])
    if str_filename:
        plt.savefig(str_filename)
    if show:
        plt.show()
    if close:
        plt.close()


def scatter_node_emb(emb_by_class, config_c, filename=None, show=True, epoch=None, ax=None, close=False, sequential_colors=False):
    num_nodi_totali = len(emb_by_class[0]) * 2 * len(emb_by_class[0][0].node_embedding_array)
    alpha_value = min(1, 3000 / num_nodi_totali)
    exps = [] #config_c.conf['graph_dataset']['list_exponents']
    custom_lines = []
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    for i, emb_class in enumerate(emb_by_class):
        color = get_colors_to_cycle_rainbow8()[i % 8]
        if sequential_colors:
            color = get_colors_to_cycle_sequential(len(emb_by_class))[i]
        custom_lines.append(Line2D([0], [0], color=color, lw=3))
        exps.append(emb_class[0].exponent)
        for emb_pergraph in emb_class:
            ax.scatter(emb_pergraph.node_label, emb_pergraph.node_embedding_array, marker='.', color=color, alpha=alpha_value)
    #for emb_pergraph in emb_perclass1:
    #    ax.scatter(emb_pergraph.node_label, emb_pergraph.node_embedding_array, marker='.', color='red', alpha=alpha_value)
    ax.set_ylabel('Node Emb. values', fontsize=16);
    ax.set_xlabel('Degree sequence', fontsize=16);
    titolo = "Node embedding vs. Degree sequence"   #f'Final Test Acc. {round(accuracy,5)}'
    if epoch:
        titolo += f'\t epoch {epoch}'
    ax.set_title(titolo)
    ax.legend(custom_lines, [f"exp {round(e,2)}" for e in exps], loc=1, prop={'size': 6})
    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=100)
    if show:
        plt.show()
    if close:
        plt.close()


def plot_node_emb_1D_perclass(emb_by_class, filename=None, show=True, sequential_colors=False):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    num_nodi_totali = len(emb_by_class) * len(emb_by_class[0]) * len(emb_by_class[0][0].node_embedding_array)
    alpha_value = min(1, 3000/num_nodi_totali)
    exps = [] #config_c.conf['graph_dataset']['list_exponents']
    custom_lines = []

    for i, emb_class in enumerate(emb_by_class):
        color = get_colors_to_cycle_rainbow8()[i % 8]
        if sequential_colors:
            color = get_colors_to_cycle_sequential(len(emb_by_class))[i]
        custom_lines.append(Line2D([0], [0], color=color, lw=3))
        exps.append(emb_class[0].exponent)
        for emb_pergraph in emb_class:
            nodeemb_sorted = sorted(emb_pergraph.node_embedding_array, reverse=True)
            degree_sorted = sorted(emb_pergraph.node_label, reverse=True)
            ax1.plot(nodeemb_sorted, '.', c=color, alpha=alpha_value, label=emb_pergraph.exponent)
            ax2.plot(degree_sorted, '.',  c=color, alpha=alpha_value, label=emb_pergraph.exponent)
            # plt.ylim(ymax = max(nodeemb_sorted), ymin = 1)

    ax1.set_title(f'Node Embedding')

    ax1.set_xlabel('idx Node Emb.', fontsize=16);
    ax1.set_ylabel('Value Node Emb.', fontsize=16);
    ax2.set_xlabel('idx node', fontsize=16);
    ax2.set_ylabel('Node degree', fontsize=16);
    ax1.legend(custom_lines, [f"exp {e}" for e in exps])
    ax2.legend(custom_lines, [f"exp {e}" for e in exps])
    # plt.yscale('log')
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

def plot_hist_node_emb_1d(emb_by_class, filename=None, show=True, sequential_colors=False):
    pass


def plot_node_emb_nD_perclass(emb_by_class, filename=None, show=True, close=False, sequential_colors=False):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    num_nodi_totali = len(emb_by_class) * len(emb_by_class[0]) * len(emb_by_class[0][0].node_embedding_array)
    alpha_value = min(1, 3000/num_nodi_totali)
    exps = [] #config_c.conf['graph_dataset']['list_exponents']
    custom_lines = []

    for i, emb_class in enumerate(emb_by_class):
        color = get_colors_to_cycle_rainbow8()[i % 8]
        if sequential_colors:
            color = get_colors_to_cycle_sequential(len(emb_by_class))[i]
        custom_lines.append(Line2D([0], [0], color=color, lw=3))
        exps.append(emb_class[0].exponent)
        for emb_pergraph in emb_class:
            #nodeemb_sorted = sorted(emb_pergraph.node_embedding_array, key=lambda x: np.linalg.norm(x), reverse=True)
            degree_sorted = sorted(emb_pergraph.node_label, reverse=True)
            a, b = emb_pergraph.node_embedding_array.T
            ax1.scatter(a,b, marker='.', color=color, alpha=alpha_value, label=emb_pergraph.exponent)
            ax2.plot(degree_sorted, '.',  color=color, alpha=alpha_value, label=emb_pergraph.exponent)
            # plt.ylim(ymax = max(nodeemb_sorted), ymin = 1)

    ax1.set_title(f'Node Embedding')

    #ax1.set_xlabel('idx Node Emb.', fontsize=16);
    #ax1.set_ylabel('Value Node Emb.', fontsize=16);
    ax2.set_xlabel('idx node', fontsize=16);
    ax2.set_ylabel('Node degree', fontsize=16);
    ax1.legend(custom_lines, [f"exp {e}" for e in exps])
    ax2.legend(custom_lines, [f"exp {e}" for e in exps])
    # plt.yscale('log')
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    if show:
        plt.show()
    if close:
        plt.close()

def plot_node_emb_3D(embeddings, config_c, filename=None, show=True, close=False, sequential_colors=False):
    node_emb_by_class = embeddings.get_all_node_emb_per_class()
    fig = plt.figure(figsize=(20, 6))
    ax = fig.add_subplot(projection='3d')
    num_gr_tipo = config_c.conf['graph_dataset']['Num_grafi_per_tipo']
    num_nodi = config_c.conf['graph_dataset']['Num_nodes']
    num_nodi_totali = len(node_emb_by_class) * num_gr_tipo * num_nodi
    alpha_value = min(1, 3000/num_nodi_totali)
    labels = [] #config_c.conf['graph_dataset']['list_exponents']
    custom_lines = []

    for i, emb_class in enumerate(node_emb_by_class):
        color = get_colors_to_cycle_rainbow8()[i % 8]
        if sequential_colors:
            color = get_colors_to_cycle_sequential(len(node_emb_by_class))[i]
        custom_lines.append(Line2D([0], [0], color=color, lw=3))
        class_label = embeddings.emb_perclass[i][0].scalar_label
        labels.append(class_label)
        for emb_pergraph in emb_class:
            a, b, c = emb_pergraph.T
            ax.scatter(a,b,c, marker='.', color=color, alpha=alpha_value, label=class_label)
            # plt.ylim(ymax = max(nodeemb_sorted), ymin = 1)

    ax.set_title(f'Node Embedding')

    #ax1.set_xlabel('idx Node Emb.', fontsize=16);
    #ax1.set_ylabel('Value Node Emb.', fontsize=16);
    ax.legend(custom_lines, [f"exp {e}" for e in labels])
    # plt.yscale('log')
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    if show:
        plt.show()
    if close:
        plt.close()


def plot_data_degree_sequence(config_c, emb_by_class, sequential_colors=False):
    custom_lines = []
    num_nodi_totali = len(emb_by_class) * len(emb_by_class[0]) * len(emb_by_class[0][0].node_embedding_array)
    alpha_value = min(1, 3000 / num_nodi_totali)

    plt.figure(figsize=(12, 6))
    exps = [] # config_c.conf['graph_dataset']['list_exponents']

    for i, emb_class in enumerate(emb_by_class):
        color = get_colors_to_cycle_rainbow8()[i % 8]
        if sequential_colors:
            color = get_colors_to_cycle_sequential(len(emb_by_class))[i]
        custom_lines.append(Line2D([0], [0], color=color, lw=3))
        exps.append(emb_class[0].exponent)
        #print(emb_class[0].exponent)
        for emb_pergraph in emb_class:
            counts = np.unique(emb_pergraph.node_label, return_counts=True)
            if sequential_colors:
                plt.loglog(*counts, c=color, alpha=alpha_value, label=emb_pergraph.exponent, linewidth=3)
            else:
                plt.loglog(*counts, c=color, alpha=alpha_value, label=emb_pergraph.exponent)

    # for emb_pergraph in emb_perclass1:
    #     counts = np.unique(emb_pergraph.node_label, return_counts=True)
    #     plt.loglog(*counts, c='red', alpha=alpha_value, label=exps[emb_pergraph.graph_label])

    # plt.legend(loc="upper left")
    plt.title(f'Node degree distribution')
    plt.xlabel('Degrees', fontsize=16);
    plt.ylabel('Number of nodes', fontsize=16);
    plt.legend(custom_lines, [f"exp {e}" for e in exps])
    # plt.gca().legend(('y0','y1'))
    plt.show()


def plot_corr_epoch(avg_corr_classes, config_c, ax=None):
    exps = config_c.conf['graph_dataset']['list_exponents']
    for i, avg_corr in enumerate(avg_corr_classes):
        ax.plot(avg_corr, label=f"exp {exps[i]}")
        ax.set_title("Correlation vs. training epochs")
        ax.set_xlabel('Epochs', fontsize=16);
        ax.set_ylabel('Correlation', fontsize=16);
        ax.set_ylim(-1,1)
        ax.legend()

def save_ffmpeg(filenameroot, outputfile):
    suppress_output = ">/dev/null 2>&1"
    os.system(f"ffmpeg -r 30 -i {filenameroot}%01d.png -vcodec mpeg4 -y {outputfile}.mp4 {suppress_output}")


def plot_ripetizioni_stesso_trial(xp, dot_key, folder):
    """
    un plot per gestire diverse occorrenze della stessa variabile
    :param xp:
    :param dot_key:
    :param folder:
    :param rootsave:
    :return:
    """
    df = xp.gc.config_dataframe
    multicolumns = tuple(dot_key.split('.'))
    try:
        variables = list(set(xp.diz_trials[dot_key]))
    except:  # allora forse non c'è bisogno di prendere i distinct
        variables = xp.diz_trials[dot_key]
    for var in variables:
        if isinstance(var, list):
            m1 = df[multicolumns] == tuple(var)
        else:
            m1 = df[multicolumns] == var
        if m1.sum() > 0:
            res = df[m1]
            cols = res.shape[0]
            fig, axs = plt.subplots(2, cols, figsize=(25, 10))
            for i, (j, row) in enumerate(res.iterrows()):
                if cols == 1:
                    ax0i = axs[0]
                    axs1i = axs[1]
                else:
                    ax0i = axs[0][i]
                    axs1i = axs[1][i]
                plt.suptitle(f"{var}")
                avg_corr_classes = row[('risultati', 'correlation_allclasses')]
                plot_corr_epoch(avg_corr_classes, xp.gc.config_class, ax0i)
                avg_tau = row[('risultati', 'tau_allclasses')]
                ax1t = ax0i.twinx()

                exps = xp.gc.config_class.conf['graph_dataset']['list_exponents']
                for k, avg_corr in enumerate(avg_tau):
                    ax1t.plot(avg_corr, label=f"exp {exps[k]}", color=get_colors_to_cycle_rainbow8()[k + 2])
                    # ax.set_title("Correlation vs. training epochs")
                    # ax.set_xlabel('Epochs', fontsize=16);
                    ax1t.set_ylabel('Kendall_Tau', fontsize=16);
                    ax1t.set_ylim(-1, 1)
                    ax1t.legend()
                # plot_corr_epoch(avg_tau, config_class, ax1t)

                axs1i.plot(row[('risultati', 'test_loss')], color=get_colors_to_cycle_rainbow8()[4])
                axs1i.set_ylabel('Test Loss', color=get_colors_to_cycle_rainbow8()[4])
                # axs1i.set_ylim(0,0.07)
                axt = axs1i.twinx()
                axt.plot(row[('risultati', 'test_accuracy')], '.', color=get_colors_to_cycle_rainbow8()[5])
                axt.set_ylabel('Accuracy', color=get_colors_to_cycle_rainbow8()[5])
                axt.set_ylim(0, 1.1)
                # axs1i.set_title(f"trial {i}")

        if folder:
            filenamesave = xp.rootsave / folder / (f'{var}.png')
            plt.savefig(filenamesave)
    plt.tight_layout()
    plt.show()


def plot_onlyloss_ripetizioni_stesso_trial(xp, dot_key, ylim=None, xlim=None, filename=None):
    df = xp.gc.config_dataframe
    multicolumns = tuple(dot_key.split('.'))
    distinte = list(set(xp.diz_trials[dot_key]))
    fig, axs = plt.subplots(len(distinte), 1, figsize=(25,50))
    k=0
    for var in set(xp.diz_trials[dot_key]):
        m1 = df[multicolumns] == var
        if m1.sum() > 0:
            res = df[m1]
            ax = axs[k]
            k+=1
            for i, (j, row) in enumerate(res.iterrows()):
                ax.plot(row[('risultati', 'test_loss')], color='black')  # get_colors_to_cycle()[i])

            ax.set_ylabel('Test Loss')  # , color=get_colors_to_cycle()[4])
            ax.set_title(f"{var}", fontsize=30)
            ax.set_xlim(0, 500)
            if ylim:
                ax.set_ylim(ylim)
            if xlim:
                ax.set_xlim(xlim)
            # axt = axs[1][i].twinx()
            # axt.plot(row[('risultati', 'test_accuracy')], '.', color=get_colors_to_cycle()[5])

    if filename:
        filenamesave = xp.rootsave / filename
        plt.savefig(filenamesave, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

def plot_onlyloss_ripetizioni_stesso_trial_superimposed(xp, dot_key, ylim=None, xlim=None, filename=None, lista_keys=None):
    df = xp.gc.config_dataframe
    multicolumns = tuple(dot_key.split('.'))
    if lista_keys is None:
        distinte = list(set(xp.diz_trials[dot_key]))
    else:
        distinte = lista_keys
    tot = len(distinte)
    custom_lines = []
    k = 0
    for var in distinte:
        colore = get_colors_to_cycle_rainbowN(tot)[k]
        custom_lines.append(Line2D([0], [0], color=colore, lw=3))
        m1 = df[multicolumns] == var
        if m1.sum() > 0:
            res = df[m1]
            for i, (j, row) in enumerate(res.iterrows()):
                plt.plot(row[('risultati', 'test_loss')], color=colore, alpha=0.5) #, label=var)
        else:
            plt.plot(0, color=colore)
        k += 1


    plt.ylabel('Test Loss')
    plt.xlim(0, 500)
    if ylim:
        plt.ylim(ylim)
    if xlim:
        plt.xlim(xlim)
    plt.legend(custom_lines, distinte)
    if filename:
        filenamesave = xp.rootsave / filename
        plt.savefig(filenamesave)

    plt.tight_layout()
    plt.show()


    # def plot_4(emb_class01, trainer, close=False):
#     for i, (emb_perclass0, emb_perclass1) in enumerate(emb_class01):
#         c = configs[i]
#         # c = config_c
#         # c.conf['last_accuracy'] = 0.99
#         # plot_data_degree_sequence(config_c, emb_perclass0, emb_perclass1)
#         # plot_node_emb_1D_perclass(emb_perclass0, emb_perclass1, c.conf['last_accuracy'])
#         fig, axes = plt.subplots(2, 2, figsize=(20, 12))
#         scatter_node_emb(emb_perclass0, emb_perclass1, c.conf['last_accuracy'], ax=axes[0][0], show=False)
#         plot_graph_emb_1D(emb_perclass0, emb_perclass1, c.conf['last_accuracy'], ax=axes[0][1], show=False)
#         axes[1][0].plot(trainer.test_loss_list)
#         axes[1][0].plot(10, trainer.test_loss_list[10], 'ro')
#         axes[1][1].plot(trainer.accuracy_list)
#         fig.suptitle(f"Esponenti power law{c.conf['graph_dataset']['list_exponents']}")
#         # plt.show()
#         if close:
#             plt.close()

# endregion
