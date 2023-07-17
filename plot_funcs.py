import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np
from scipy import stats
from plt_parameters import get_colors_to_cycle_rainbow8, get_colors_to_cycle_rainbowN, get_colors_to_cycle_sequential
from matplotlib.lines import Line2D
from config_valid import TrainingMode, GraphType
import umap
from matplotlib import ticker
from cycler import cycler
from utils_embednet import array_wo_outliers
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1,1))

sc1 = None
class Data2Plot():
    """
    Serve per generare una generica classe di dati da plottare, andando a specificare
    di volta in volta quali data plottare in questa classe, invece della funzione di plot
    Gerarchia:
        - oggetto contenente i dati
        - classi diverse di training
        - graph_embedding o final output: un qualsiasi dato che descrive il grafo per intero
        - node embedding: un array per ogni grafo che descrive i nodi
    """
    def __init__(self, input_obj, dim, config_class=None):
        self.config_class = config_class
        self.input_obj = input_obj
        print(f"Shape input object: {self.input_obj.shape}")
        self.array2plot = []
        self.class_labels = []
        self.unique_class_labels = []
        self.dim = dim

        try:
            self.allgraph_class_labels = [[emb.scalar_label for emb in emb_per_graph] for emb_per_graph in self.input_obj]
            if config_class:
                if config_class.graphtype == GraphType.SBM:  #problema perché ho un'etichetta che è una matrice
                    mapdict = {str(c): i for i, c in enumerate(config_class.conf['graph_dataset']['community_probs'])}
                    self.allgraph_class_labels = np.array([[mapdict[str(emb.scalar_label)] for emb in emb_per_graph] for emb_per_graph in self.input_obj])
            self.allnode_degree = [[emb.actual_node_class for emb in emb_per_graph] for emb_per_graph in self.input_obj]
            self.allnode_labels_scalar_class = [[[emb.scalar_label]*len(emb.actual_node_class) for emb in emb_per_graph] for emb_per_graph in self.input_obj]
            #self.set_data()
        except Exception as e:
            print(f"Warning: problema forse con autoencoder 2 plot \n {e}")
            pass

    def set_data(self, type=None):
        self.array2plot = []
        self.class_labels = []
        if type is None:
            raise Exception("Must set a data type")

        if type == 'graph_embedding':
            for classe in self.input_obj:
                self.class_labels.append(classe[0].scalar_label)
                arr = []
                for graph in classe:
                    arr.append(graph.graph_embedding)
                self.array2plot.append(arr)

        elif type == 'node_embedding':
            for classe in self.input_obj:
                self.class_labels.append(classe[0].scalar_label)
                arr = []
                for graph in classe:
                    arr.append(graph.node_embedding_array)
                self.array2plot.append(arr)

        elif type == 'final_output':
            for classe in self.input_obj:
                self.class_labels.append(classe[0].scalar_label)
                arr = []
                for graph in classe:
                    arr.append(graph.output)
                self.array2plot.append(arr)

        self.array2plot = np.squeeze(np.array(self.array2plot))
        self.unique_class_labels = list(set(self.class_labels))

    def get_color(self, i, sequential_colors=False):
        if sequential_colors:
            color = get_colors_to_cycle_sequential(len(self.input_obj))[i]
        else:
            color = get_colors_to_cycle_rainbow8()[i % 8]
        return color
    def get_alpha_value(self, type):
        num_tot = 0
        if type == 'node_embedding':
            num_tot = len(self.input_obj) * len(self.input_obj[0]) * len(self.input_obj[0][0].node_embedding_array)
        elif type == 'graph_embedding' or type == 'final_output':
            num_tot = len(self.input_obj) * len(self.input_obj[0])
        elif type == 'adj_entries':
            num_tot = len(self.input_obj)# * len(self.input_obj[0])
        alpha_value = min(1, 2000 / num_tot)
        return alpha_value

    def plot(self, datatype='node_embedding', type='histogram',
             ax=None, filename=None, ylim=None, xlim=None,
             sequential_colors=False, log=False, title=None):

        self.set_data(datatype)
        if self.array2plot is None:  #.all():
            return



        if datatype == 'adj_entries':
            self.plot_adj_entries_hist(ax)

        else:
            ## caso in cui devo fare un umap e cmq devo plottare tutte le dimensioni dell'embedding
            if self.dim > 2:
                #print(f"shape array:  {self.array2plot.shape}")
                # voglio appiattire ma mantenere la suddivisione per classe
                # array2plotflattened = self.array2plot.reshape(-1, self.array2plot.shape[-1])
                array2plotflattened = self.array2plot.reshape(self.array2plot.shape[0], -1, self.array2plot.shape[-1])
                #emb_data = self.calc_umap(array2plotflattened)
                #if datatype == 'node_embedding':
                    #print(f"labels ripetute pergraph prima: {np.array(self.allnode_labels_scalar_class).shape}")
                    #labels_ripetute_pergraph = np.array(self.allnode_labels_scalar_class).flatten()
                #elif datatype == 'graph_embedding' or datatype == 'final_output':
                    #print(f"labels ripetute pergraph prima: {np.array(self.allgraph_class_labels).shape}")
                    #labels_ripetute_pergraph = np.array(self.allgraph_class_labels).flatten()
                #print(f"labels ripetute pergraph dopo: {labels_ripetute_pergraph.shape}")
                #cmap = mpl.colors.LinearSegmentedColormap.from_list("", get_colors_to_cycle_rainbow8())
                #cmap = mpl.colors.ListedColormap(get_colors_to_cycle_sequential(len(self.input_obj)), name='from_list', N=None)
                ## PLOT now:
                #ax.scatter(emb_data[:, 0], emb_data[:, 1], c=labels_ripetute_pergraph, cmap=cmap, alpha=alpha_value) #'gist_rainbow');

                # lascio stare l'UMAP al momento e uso il parallel coord plot
                custom_lines = []
                for i, classe in enumerate(array2plotflattened):
                    color = self.get_color(i, True)
                    custom_lines.append(Line2D([0], [0], color=color, lw=3))
                    label = str(self.unique_class_labels[i])
                    alpha_value = self.get_alpha_value(datatype, i)
                    parallel_coord(classe, ax, title, color=color, label=label, alpha_value=alpha_value)

                ax.legend(custom_lines, [f"class {e}" for e in self.unique_class_labels])

            ## devo plottare:
            ## node emb come scatter
            ## graph emb come histogram
            ## l'output sarà sempre minimo a 2D , però quì devo richiamarlo comunue quando ho un embedding 1D
            else:
                if ax is None:
                    fig, ax = plt.subplots(figsize=(12, 6))
                custom_lines = []
                for i, classe in enumerate(self.array2plot):
                    color = self.get_color(i, sequential_colors)
                    custom_lines.append(Line2D([0], [0], color=color, lw=3))
                    label = str(self.unique_class_labels[i])
                    if type == 'histogram':
                        self.plot_hist(ax, classe, color, label)
                    elif type == 'plot':
                        ax.plot(*classe.T, c=color, alpha=alpha_value, label=label, marker='.', linestyle='None')
                    elif type == 'scatter':
                        ## dim 1 scatter: abbiamo il node embedding VS. node degree
                        if log:
                            ax.plot(np.log10(np.array(self.allgraph_class_labels)), np.log10(classe + 1), marker='.', linestyle='None', color=color, alpha=alpha_value)
                        else:
                            ax.plot(self.all_node_degree[i], classe, marker='.', linestyle='None', color=color, alpha=alpha_value)

                ax.legend(custom_lines, [f"class {e}" for e in self.unique_class_labels])

        if title:
            ax.set_title(title)
        if filename:
            plt.savefig(filename, bbox_inches='tight')
        if ylim:
            ax.set_ylim(ylim)
        if xlim:
            ax.set_xlim(xlim)

    def plot_hist(self, ax, embedding, color, label):
        """
        :param ax:
        :param embedding: può essere una classe di embeddings
        :param color:
        :param label:
        :return:
        """
        # counts = np.unique(classe, return_counts=True)
        # if log:
        #    ax.loglog(*counts, c=color, alpha=alpha_value, label=self.class_labels[i], linewidth=3)
        # else:
        #    ax.plot(*counts, c=color, alpha=alpha_value, label=self.class_labels[i], linewidth=3)
        new_bins = np.histogram(np.hstack(self.array2plot), bins=30)[1]
        h, e = np.histogram(embedding, bins=new_bins)  # , density=density)
        ax.bar(e[:-1], h, width=np.diff(e), ec='k', align='edge', label=label, color=color, alpha=0.7)
        # ax.hist(classe.flatten(), color=color, label=self.class_labels[i], bins=30)



    @classmethod
    def static_plot_for_animation(cls, array2plot, class_labels, datatype='node_embedding', type='histogram',
             filename=None, ylim=None, xlim=None,
             sequential_colors=False, log=False, title=None):

        #sc1.set_data(x, y)
        custom_lines = []
        if datatype == 'node_embedding':
            num_tot = len(array2plot) * len(array2plot[0]) * len(array2plot[0][0])
        elif datatype == 'graph_embedding' or datatype == 'final_output':
            num_tot = len(array2plot) * len(array2plot[0])
        alpha_value = min(1, 3000 / num_tot)

        for i, classe in enumerate(array2plot):
            if sequential_colors:
                color = get_colors_to_cycle_sequential(len(array2plot))[i]
            else:
                color = get_colors_to_cycle_rainbow8()[i % 8]
            custom_lines.append(Line2D([0], [0], color=color, lw=3))
            sc1.set_data(*classe.T)#, color=color, alpha=alpha_value, label=class_labels[i], marker='.', linestyle='None')
            sc1.set_color(color)
        #   sc1.set_data(x, y*i)
        return sc1

    def calc_umap(self, data):
        print("UMAP...")
        #print(f"shape array: {data.shape}")
        embedding = umap.UMAP(n_epochs=10, n_neighbors=10, n_jobs=32, verbose=False).fit_transform(data)
        return embedding


class DataAutoenc2Plot(Data2Plot):
    """

    """
    def __init__(self, wrapper_obj, dim, config_class=None, sequential_colors=False, metric_name='da_impostare'):
        self.config_class = config_class
        self.wrapper_obj = wrapper_obj
        self.input_obj = wrapper_obj.list_emb_autoenc_per_graph
        self.array2plot = []
        ## la label di classe è prodotta nel training set
        self.class_labels = [graph.scalar_label for graph in self.input_obj]
        self.unique_class_labels = list(set(self.class_labels))
        ## la label di nodo può essere diversa per ogni nodo (actual node degree  oppure original node label)
        ## oppure posso voler associare a ogni nodo la label del grafo a cui il nodo appartiene
        self.allnode_labels = None ## [graph.node_label_from_dataset for graph in self.input_obj]
        self.all_node_degree = None
        self.dim = dim
        self.colors = None
        self.custom_legend_lines = None
        self.sequential_colors = sequential_colors
        self.metric_name = metric_name
        #super().__init__(input_obj, dim, config_class)
        #print("Chiamo DataAutoenc2Plot: inizializzato correttamente.")

    def set_data(self, type=None):
        self.array2plot = []

        if type == 'adj_entries':
            outputs = []
            inputs = []
            adj_01 = []
            for graph in self.input_obj:
                outputs.append(graph.decoder_output.ravel())
                inputs.append(graph.input_adj_mat.ravel())
                adj_01.append(graph.thresholded.ravel())
            self.array2plot = (np.array(inputs), np.array(outputs), np.array(adj_01))

        elif type == 'node_embedding':
            self.array2plot = np.array([[graph.node_embedding.squeeze() for graph in self.input_obj if graph.scalar_label == classe] for classe in self.unique_class_labels])
            self.allnode_labels = [[graph.node_label_from_dataset for graph in self.input_obj if graph.scalar_label == classe] for classe in self.unique_class_labels]
            self.all_node_degree = [[graph.node_degree for graph in self.input_obj if graph.scalar_label == classe] for classe in self.unique_class_labels]
            self.colors = {self.unique_class_labels[c]: self.get_color(c, self.sequential_colors) for c in range(len(self.unique_class_labels))}
            self.custom_legend_lines = [(Line2D([0], [0], color=color, lw=3)) for color in self.colors.values()]

        elif type == 'graph_embedding':
            for graph in self.input_obj:
                graph.calc_graph_emb()
            self.array2plot = np.array([[graph.graph_embedding.squeeze() for graph in self.input_obj if graph.scalar_label == classe] for classe in self.unique_class_labels])

    def plot_adj_entries_hist(self, ax):
        input_adj_flat, pred_adj_flat, pred_after_sigm = self.array2plot
        # input_adj_flat, pred_adj_flat = input_adj_flat[0], pred_adj_flat[0]
        input_adj_flat, pred_adj_flat, pred_after_sigm_flat = input_adj_flat.ravel(), pred_adj_flat.ravel(), pred_after_sigm.ravel()
        ax.hist((input_adj_flat, pred_adj_flat, pred_after_sigm_flat), bins=50, range=[-0.25, 1.25]);

    def get_color(self, i, sequential_colors=False):
        if sequential_colors:
            color = get_colors_to_cycle_sequential(len(self.array2plot))[i]
        else:
            color = get_colors_to_cycle_rainbow8()[i % 8]
        return color
    def get_alpha_value(self, type, i=None):
        num_tot = 0
        if type == 'node_embedding':
            num_tot = len(self.input_obj) * len(self.input_obj[0].node_embedding)
        elif type == 'graph_embedding' or type == 'final_output':
            num_tot = len(self.input_obj)
        elif type == 'adj_entries':
            num_tot = len(self.input_obj)# * len(self.input_obj[0])
        alpha_value = min(1, 500 / num_tot)
        if i is not None:
            arr = [3000, 2000, 1000, 500, 200, 100, 50]
            alpha_value = min(1, arr[min(6,i)] / num_tot)
        #print(f"num_tot per alpha value: {num_tot}  - alpha: {alpha_value}")
        return alpha_value


    # def plot(self, datatype='node_embedding', type='histogram',
    #          ax=None, filename=None, ylim=None, xlim=None,
    #          sequential_colors=False, log=False, title=None):
    #
    #     self.set_data(datatype)
    #     alpha_value = self.get_alpha_value(datatype)
    #
    #     if datatype == 'adj_entries':
    #         self.plot_adj_entries_hist(ax)
    #     else:
    #         ## TODO:: modificare per embedding autoencoder
    #         if self.dim > 2:
    #             #print(f"shape array:  {self.array2plot.shape}")
    #             array2plotflattened = self.array2plot.reshape(-1, self.array2plot.shape[-1])
    #             emb_data = self.calc_umap(array2plotflattened)
    #             if datatype == 'node_embedding':
    #                 #print(f"labels ripetute pergraph prima: {np.array(self.allnode_labels_scalar_class).shape}")
    #                 labels_ripetute_pergraph = np.array(self.allnode_labels_scalar_class).flatten()
    #             elif datatype == 'graph_embedding':
    #                 #print(f"labels ripetute pergraph prima: {np.array(self.allgraph_class_labels).shape}")
    #                 labels_ripetute_pergraph = np.array(self.allgraph_class_labels).flatten()
    #             #print(f"labels ripetute pergraph dopo: {labels_ripetute_pergraph.shape}")
    #             cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", get_colors_to_cycle_rainbow8())
    #             #cmap = matplotlib.colors.ListedColormap(get_colors_to_cycle_sequential(len(self.input_obj)), name='from_list', N=None)
    #             ## PLOT now:
    #             ax.scatter(emb_data[:, 0], emb_data[:, 1], c=labels_ripetute_pergraph, cmap=cmap, alpha=alpha_value) #'gist_rainbow');
    #         else:
    #             for i, emb_pergraph in enumerate(self.array2plot):
    #                 color = self.colors[self.class_labels[i]]
    #                 label = str(self.class_labels[i])
    #
    #                 if type == 'scatter':
    #                     ax.plot(self.allnode_labels[i], emb_pergraph, marker='.', linestyle='None', color=color, alpha=alpha_value)
    #                 elif type == 'histogram':
    #                     self.plot_hist(ax, emb_pergraph, color, label)
    #
    #             ax.legend(self.custom_legend_lines, [f"class {e}" for e in self.class_labels])




# region plot functions
def plot_metrics(data, num_emb_neurons, test_loss_list=None, current_metric_list=None,
                 epochs_list=None,
                 node_intrinsic_dimensions_total=None, graph_intrinsic_dimensions_total=None,
                 model_pars=None, param_labels=None,
                 node_correlation=None, graph_correlation=None,
                 sequential_colors=False, log=False, **kwargs):


    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    if num_emb_neurons == 1:
        #print("Plotting 1D embeddings...")
        data.plot(datatype='node_embedding', type='scatter', ax=axes[0][0], sequential_colors=sequential_colors, title="Node Embedding")
        data.plot(datatype='graph_embedding', type='histogram', ax=axes[0][1], sequential_colors=sequential_colors, title="Graph Embedding")
        # TODO: rimettere
        #plot_node_and_graph_correlations(axes, graph_correlation, node_correlation, epochs_list, **kwargs)
    else:
        #print("Plotting 2D or n>=2 embeddings...")
        #if num_emb_neurons < 3:     # per evitare di calcolare UMAP per ogni frame   #kwargs.get('plot_node_embedding', True):
        data.plot(datatype='node_embedding', type='plot', ax=axes[0][0], sequential_colors=sequential_colors, title="Node Embedding")
        data.plot(datatype='graph_embedding', type='plot', ax=axes[0][1], sequential_colors=sequential_colors, title="Graph Embedding")
        if node_intrinsic_dimensions_total is not None and graph_intrinsic_dimensions_total is not None:
            pass
            # TODO: rimettere
            #plot_intrinsic_dimension(axes, graph_intrinsic_dimensions_total, node_intrinsic_dimensions_total, epochs_list, **kwargs)

    if data.config_class.autoencoding:
        data.plot(datatype='adj_entries', type='hist', ax=axes[0][2], sequential_colors=sequential_colors, title="Adj matrix entries")
    else:
        data.plot(datatype='final_output', type='plot', ax=axes[0][2], sequential_colors=sequential_colors, title="Final Output")


    plot_test_loss_and_metric(axes, current_metric_list, test_loss_list, epochs_list, **kwargs)

    plot_weights_multiple_hist(model_pars, param_labels, axes[1][2], absmin=-2, absmax=2, sequential_colors=False)

    fig.suptitle(f"{kwargs['long_string_experiment']}")

    if kwargs['showplot']:
        plt.show()
    return fig


def plot_test_loss_and_metric(axes, current_metric_list, test_loss_list, epochs_list, **kwargs):
    x_max = kwargs.get("last_epoch")
    # test loss
    loss_list_min, loss_list_max = min(array_wo_outliers(test_loss_list)), max(array_wo_outliers(test_loss_list))
    #loss_list_min, loss_list_max = min(test_loss_list), max(test_loss_list)
    ploss, = axes[1][0].plot(epochs_list, test_loss_list, color='black', label='Test Loss')
    axes[1][0].set_ylim(loss_list_min, loss_list_max)
    axes[1][0].set_xlim(0, x_max)
    # axes[1][0].set_ylabel('Test Loss', fontsize=12);
    axes[1][0].legend(loc='lower left')
    # axes[1][0].yaxis.label.set_color(ploss.get_color())

    # plot accuracy
    # if not (data.config_class.conf['model'].get('autoencoder') or data.config_class.conf['model'].get('autoencoder_confmodel')):
    axt = axes[1][0].twinx()
    pmetric, = axt.plot(epochs_list, current_metric_list, color='blue', label=kwargs.get("metric_name"))
    axt.set_ylim(0, 1)
    # axt.set_ylabel('Test metric', fontsize=12);
    axt.set_xlim(0, x_max)
    #    axt.set_yticklabels([0.0,1.0])
    axt.legend(loc='upper right')
    # axt.yaxis.label.set_color(pmetric.get_color())
    axt.tick_params(axis='y', colors=pmetric.get_color())


def plot_intrinsic_dimension(axes, graph_intrinsic_dimensions_total, node_intrinsic_dimensions_total, epochs_list, **kwargs):
    x_max = kwargs.get("last_epoch")
    axes[1][1].plot(epochs_list, node_intrinsic_dimensions_total, linestyle='None', marker='.', color='red', label='node id')
    axes[1][1].plot(epochs_list, graph_intrinsic_dimensions_total, linestyle='None', marker='.', color='blue', label='graph id')
    axes[1][1].set_xlim(0, x_max)
    axes[1][1].set_ylim(0, 3.0)
    axes[1][1].set_title(f"Intrinsic Dimensionality")
    axes[1][1].legend()


def plot_node_and_graph_correlations(axes, graph_correlation, node_correlation, epochs_list, **kwargs):
    x_max = kwargs.get("last_epoch")
    axes[1][1].plot(epochs_list, node_correlation, linestyle='None', marker='.', color='red', label='Node Correlation')
    axes[1][1].plot(epochs_list, graph_correlation, linestyle='None', marker='.', color='blue', label='Graph Correlation')
    axes[1][1].set_xlim(0, x_max)
    axes[1][1].set_title(f"Embedding corr - degree sequence")
    axes[1][1].set_ylim(-1.0, 1.0)


def parallel_coord(ys, host, titolo_grafico, **kwargs):
    """
    from https://stackoverflow.com/questions/8230638/parallel-coordinates-plot-in-matplotlib
    :param ys: array da visualizzare
    """
    #array = np.array([np.random.normal(mu, 0.5, 16) for mu in np.arange(-3, 3, 0.01)])
    #ys = array
    #print(ys.shape)
    ymins = ys.min(axis=0)
    ymin_abs = ymins.min()
    ymaxs = ys.max(axis=0)
    ymax_abs = ymaxs.max()
    dys = ymaxs - ymins
    dys_abs = ymax_abs - ymin_abs
    ymins -= dys * 0.05  # add 5% padding below and above
    ymaxs += dys * 0.05
    ymin_abs -= dys_abs * 0.05
    ymax_abs += dys_abs * 0.05

    dys = ymaxs - ymins
    dys_abs = ymax_abs - ymin_abs

    # transform all data to be compatible with the main axis
    zs = np.zeros_like(ys)
    zs[:, 0] = ys[:, 0]
    zs[:, 1:] = (ys[:, 1:] - ymins[1:]) / dys[1:] * dys[0] + ymins[0]

    #fig, host = plt.subplots(figsize=(10,4))

    axes = [host] + [host.twinx() for i in range(ys.shape[1] - 1)]

    for i, ax in enumerate(axes):
        #ax.set_ylim(ymins[i], ymaxs[i])
        ax.set_ylim(ymin_abs, ymax_abs)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        if ax != host:
            ax.set_axis_off()
            ax.spines['left'].set_visible(False)
            #ax.yaxis.set_ticks_position('right')
            ax.spines["right"].set_position(("axes", i / (ys.shape[1] - 1)))  #.set_visible(False)
            ax.yaxis.set_major_locator(ticker.NullLocator())

    host.set_xlim(0, ys.shape[1] - 1)
    host.set_xticks(range(ys.shape[1]))
    #host.set_xticklabels(ynames, fontsize=14)
    host.tick_params(axis='x', which='major', pad=7)
    host.spines['right'].set_visible(False)
    #host.xaxis.tick_top()
    host.set_title(titolo_grafico, fontsize=16, pad=12)

    #colors = plt.cm.Set2.colors
    # for j in range(ys.shape[0]):
    #     # create bezier curves
    #     verts = list(zip([x for x in np.linspace(0, len(ys) - 1, len(ys) * 3 - 2, endpoint=True)],
    #                      np.repeat(zs[j, :], 3)[1:-1]))
    #     codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]
    #     path = Path(verts, codes)
    #     color = get_colors_to_cycle_sequential(len(ys))[j]
    #     patch = patches.PathPatch(path, facecolor='none', lw=2, alpha=0.3, edgecolor=color)
    #     #legend_handles[iris.target[j]] = patch
    #     host.add_patch(patch)

    #cmap = mpl.colors.LinearSegmentedColormap.from_list("", get_colors_to_cycle_rainbow8())
    #custom_cycler = (cycler(color=get_colors_to_cycle_sequential(len(ys))))
    #host.set_prop_cycle(custom_cycler)
    host.plot(ys.T,  lw=2, alpha=kwargs.get("alpha_value"), color=kwargs.get("color"), label=kwargs.get("label"))  #, cmap=cmap)
    host.plot(ys.T, '.', alpha=kwargs.get("alpha_value"), color=kwargs.get("color")) #, alpha=0.8)


    #host.legend(legend_handles, iris.target_names,
    #            loc='lower center', bbox_to_anchor=(0.5, -0.18),
    #            ncol=len(iris.target_names), fancybox=True, shadow=True)
    #plt.tight_layout()
    return

# endregion

# region old plot functions
def plot_dim1(embeddings_per_class, bins=30, want_kde=True, density=True, nomefile=None, labels=None, title=None, sequential_colors=False):
    #plt.figure(figsize=(18, 6))  # , dpi=60)
    # trova lo stesso binning
    new_bins = np.histogram(np.hstack(embeddings_per_class), bins=bins)[1]

    for i, emb in enumerate(embeddings_per_class):
        h, e = np.histogram(emb, bins=new_bins, density=density)
        x = np.linspace(e.min(), e.max())
        if labels:
            lab = labels[i]
            color = get_colors_to_cycle_rainbow8()[i % 8]
            if sequential_colors:
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



def plot_graph_emb_1D(emb_by_class, str_filename=None, show=True, ax=None, close=False, sequential_colors=False):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    scalar_label = []

    for i, emb_class in enumerate(emb_by_class):
        hist = []
        scalar_label.append(emb_class[0].scalar_label)
        for emb_pergraph in emb_class:
            hist.append(emb_pergraph.graph_embedding)
        hist = np.array(hist).flatten()
        if sequential_colors:
            color = get_colors_to_cycle_sequential(len(emb_by_class))[i]
            ax.hist(hist, bins=30, label=f"exp {scalar_label[i]}", color=color);
        else:
            ax.hist(hist, bins=30, label=f"exp {scalar_label[i]}")

    ax.set_title(f'Graph Embedding')
    ax.legend(loc=1, prop={'size': 8})  #custom_lines, [f"exp {e}" for e in exps])
    if str_filename:
        plt.savefig(str_filename)
    if show:
        plt.show()
    if close:
        plt.close()


def plot_graph_emb_nD(emb_by_class, filename=None, ax=None, show=True, close=False, sequential_colors=False, d=2):
    if ax is None:
        fig = plt.figure(figsize=(12, 6))
        if d == 2:
            ax = fig.add_subplot()#projection='3d')
        elif d == 3:
            ax = fig.add_subplot(projection='3d')
    #num_nodi_totali = len(emb_by_class) * len(emb_by_class[0]) * len(emb_by_class[0][0].node_embedding_array)
    #alpha_value = min(1, 3000/num_nodi_totali)
    labels = [] #config_c.conf['graph_dataset']['list_exponents']
    #custom_lines = []
    #graph_emb_perclass = embeddings.get_all_graph_emb_per_class()
    # if graph_emb_perclass.ndim == 2:  # non è suddiviso per classe e quindi non plot color
    #     a, b, c = embedding.T
    #     ax.scatter(a, b, c)
    # graph_emb_perclass.ndim == 3:  # ho le classi
    for i, emb_class in enumerate(emb_by_class):
        sc = []
        labels.append(emb_class[0].scalar_label)
        for emb_pergraph in emb_class:
            sc.append(emb_pergraph.graph_embedding)
        sc = np.array(sc)#.flatten()
        ax.plot(*sc.T, label=f"{emb_class[0].scalar_label}", marker='.', linestyle='None')  # alpha=alpha_value,

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


def scatter_node_emb(emb_by_class, filename=None, show=True, epoch=None, ax=None, close=False, sequential_colors=False, labels=True, xlim=None, ylim=None, log=False):
    """
    Plotta il node embedding rispetto a una label di classe vettoriale, va bene nel caso della power law
    dove c'è una label per ogni nodo
    :return:
    """
    num_nodi_totali = len(emb_by_class[0]) * 2 * len(emb_by_class[0][0].node_embedding_array)
    alpha_value = min(1, 3000 / num_nodi_totali)
    exps = []
    custom_lines = []
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    for i, emb_class in enumerate(emb_by_class):
        color = get_colors_to_cycle_rainbow8()[i % 8]
        if sequential_colors:
            color = get_colors_to_cycle_sequential(len(emb_by_class))[i]
        custom_lines.append(Line2D([0], [0], color=color, lw=3))
        lab = emb_class[0].scalar_label
        exps.append(lab)
        for emb_pergraph in emb_class:
            if log:
                ax.plot(np.log10(np.array(emb_pergraph.actual_node_class)), np.log10(emb_pergraph.node_embedding_array + 1), marker='.', linestyle='None', color=color, alpha=alpha_value)
                #ax.yaxis.set_major_formatter(formatter)
            else:
                ax.plot(emb_pergraph.actual_node_class, emb_pergraph.node_embedding_array, marker='.', linestyle='None', color=color, alpha=alpha_value)

    if ylim:
        ax.set_ylim(ylim)
    if xlim:
        ax.set_xlim(xlim)
    # if log:
    #     ax.set_yscale('log')
    #     plt.yscale('log')
    if labels:
        ax.set_ylabel('Node Emb. values', fontsize=16);
        ax.set_xlabel('Degree sequence', fontsize=16);
        titolo = "Node embedding vs. Degree sequence"   #f'Final Test Acc. {round(accuracy,5)}'
        if epoch:
            titolo += f'\t epoch {epoch}'
        ax.set_title(titolo)
    else:
        plt.setp(ax.get_yticklabels(), visible=False)
    ax.legend(custom_lines, [f"Label {e}" for e in exps], loc=1, prop={'size': 7})
    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=200)
    if show:
        plt.show()
    if close:
        plt.close()


def plot_node_emb_1D(emb_by_class, filename=None, show=True, sequential_colors=False, log=False):
    num_nodi_totali = len(emb_by_class) * len(emb_by_class[0]) * len(emb_by_class[0][0].node_embedding_array)
    alpha_value = min(1, 3000/num_nodi_totali)

    scalar_label = []
    custom_lines = []
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    for i, emb_class in enumerate(emb_by_class):
        color = get_colors_to_cycle_rainbow8()[i % 8]
        if sequential_colors:
            color = get_colors_to_cycle_sequential(len(emb_by_class))[i]
        custom_lines.append(Line2D([0], [0], color=color, lw=3))
        scalar_label.append(emb_class[0].scalar_label)
        for emb_pergraph in emb_class:
            counts = np.unique(emb_pergraph.node_embedding_array, return_counts=True)
            nodeemb_sorted = sorted(emb_pergraph.node_embedding_array, reverse=True)
            if log:
                ax1.loglog(*counts, c=color, alpha=alpha_value, label=emb_pergraph.scalar_label, linewidth=3)
                ax2.loglog(nodeemb_sorted, '.', c=color, alpha=alpha_value, label=emb_pergraph.scalar_label)
            else:
                ax1.plot(*counts, c=color, alpha=alpha_value, label=emb_pergraph.scalar_label, linewidth=3)
                ax2.plot(nodeemb_sorted, '.', c=color, alpha=alpha_value, label=emb_pergraph.scalar_label)

            # plt.ylim(ymax = max(nodeemb_sorted), ymin = 1)

    ax1.set_title(f'Node Embedding distribution')
    ax1.set_xlabel('Node Emb. values', fontsize=16);
    ax1.set_ylabel('Number of nodes', fontsize=16);
    ax1.legend(custom_lines, [f"class {e}" for e in scalar_label])

    ax2.set_title(f'Ordered Node Embedding')
    ax2.set_xlabel('idx node', fontsize=16);
    ax2.set_ylabel('Value Node Emb.', fontsize=16);
    ax2.legend(custom_lines, [f"class {e}" for e in scalar_label])
    # plt.yscale('log')
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


def plot_node_emb_nD(emb_by_class, filename=None, show=True, close=False, ax=None, sequential_colors=False):
    num_nodi_totali = len(emb_by_class) * len(emb_by_class[0]) * len(emb_by_class[0][0].node_embedding_array)
    alpha_value = min(1, 3000/num_nodi_totali)

    scalar_label = []
    custom_lines = []
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    for i, emb_class in enumerate(emb_by_class):
        # quì con [::-1] ho invertito l'ordine delle classi per un plot più chiaro
        color = get_colors_to_cycle_rainbow8()[i % 8]
        if sequential_colors:
            color = get_colors_to_cycle_sequential(len(emb_by_class))[i]
        custom_lines.append(Line2D([0], [0], color=color, lw=3))
        scalar_label.append(emb_class[0].scalar_label)
        for emb_pergraph in emb_class:
            ax.plot(*emb_pergraph.node_embedding_array.T, marker='.', color=color, alpha=alpha_value, label=emb_pergraph.scalar_label, linestyle='None')
            # plt.ylim(ymax = max(nodeemb_sorted), ymin = 1)

    ax.set_title(f'Node Embedding')
    ax.set_xlabel('dim 1', fontsize=16);
    ax.set_ylabel('dim 2', fontsize=16);
    ax.legend(custom_lines, [f"exp {e}" for e in scalar_label])
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


def plot_data_degree_sequence(emb_by_class, sequential_colors=False, log=False):
    num_nodi_totali = len(emb_by_class) * len(emb_by_class[0]) * len(emb_by_class[0][0].node_embedding_array)
    alpha_value = min(1, 3000 / num_nodi_totali)

    scalar_label = []
    custom_lines = []
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    for i, emb_class in enumerate(emb_by_class):
        color = get_colors_to_cycle_rainbow8()[i % 8]
        if sequential_colors:
            color = get_colors_to_cycle_sequential(len(emb_by_class))[i]
        custom_lines.append(Line2D([0], [0], color=color, lw=3))
        scalar_label.append(emb_class[0].scalar_label)
        for emb_pergraph in emb_class:
            counts = np.unique(emb_pergraph.actual_node_class, return_counts=True)
            degree_sorted = sorted(emb_pergraph.actual_node_class, reverse=True)
            #if sequential_colors:
            if log:
                ax1.loglog(*counts, c=color, alpha=alpha_value, label=emb_pergraph.scalar_label, linewidth=3)
                ax2.loglog(degree_sorted, '.', c=color, alpha=alpha_value, label=emb_pergraph.scalar_label)
            else:
                ax1.plot(*counts, c=color, alpha=alpha_value, label=emb_pergraph.scalar_label, linewidth=3)
                ax2.plot(degree_sorted, '.', c=color, alpha=alpha_value, label=emb_pergraph.scalar_label)
            #else:
            #    if log:
            #        ax1.loglog(*counts, c=color, alpha=alpha_value, label=emb_pergraph.scalar_label)
            #    else:
            #        ax1.plot(*counts, c=color, alpha=alpha_value, label=emb_pergraph.scalar_label)

    # plt.legend(loc="upper left")
    ax1.set_title(f'Node degree distribution')
    ax1.set_xlabel('Degrees', fontsize=16);
    ax1.set_ylabel('Number of nodes', fontsize=16);
    ax1.legend(custom_lines, [f"Label {e}" for e in scalar_label])

    ax2.set_title(f'Ordered node degree values')
    ax2.set_xlabel('idx node', fontsize=16);
    ax2.set_ylabel('Node degree', fontsize=16);
    ax2.legend(custom_lines, [f"class {e}" for e in scalar_label])
    # plt.gca().legend(('y0','y1'))
    plt.show()

def plot_recon_degree_sequence(recon_graphs):
    degs = []
    for gr in recon_graphs:
        degs.append(gr.degree())

def plot_corr_epoch(avg_corr_classes, config_c, ax=None):
    exps = config_c.conf['graph_dataset']['list_exponents']
    for i, avg_corr in enumerate(avg_corr_classes):
        ax.plot(avg_corr, label=f"exp {exps[i]}")
        ax.set_title("Correlation vs. training epochs")
        ax.set_xlabel('Epochs', fontsize=16);
        ax.set_ylabel('Correlation', fontsize=16);
        ax.set_ylim(-1,1)
        ax.legend()

# endregion

def save_ffmpeg(filenameroot, outputfile):
    suppress_output = ">/dev/null 2>&1"
    os.system(f"ffmpeg -r 30 -i {filenameroot}%01d.png -vcodec mpeg4 -y {outputfile}.mp4 {suppress_output}")


# region plot results of grid search

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


def plot_onlyloss_ripetizioni_stesso_trial_xp(xp, dot_key, ylim=None, xlim=None, figsize=(25,50), filename=None):
    df = xp.gc.config_dataframe
    diz_trials = xp.diz_trials
    rootsave = xp.rootsave
    plot_onlyloss_ripetizioni_stesso_trial(df, diz_trials, dot_key, figsize, rootsave, filename, xlim, ylim)

def plot_onlyloss_ripetizioni_stesso_trial(df, diz_trials, dot_key, figsize=(25,50), rootsave=None, filename=None, xlim=None, ylim=None):
    multicolumns = tuple(dot_key.split('.'))
    distinte = list(set(diz_trials[dot_key]))
    fig, axs = plt.subplots(len(distinte), 1, figsize=figsize)
    k = 0
    for var in set(diz_trials[dot_key]):
        m1 = df[multicolumns] == var
        if m1.sum() > 0:
            res = df[m1]
            ax = axs[k]
            k += 1
            for i, (j, row) in enumerate(res.iterrows()):
                ax.plot(row[('risultati', 'test_loss')], color='black')  # get_colors_to_cycle()[i])

            ax.set_ylabel('Test Loss')  # , color=get_colors_to_cycle()[4])
            ax.set_title(f"{var}", fontsize=30)
            #ax.set_xlim(0, 500)
            if ylim:
                ax.set_ylim(ylim)
            if xlim:
                ax.set_xlim(xlim)
            # axt = axs[1][i].twinx()
            # axt.plot(row[('risultati', 'test_accuracy')], '.', color=get_colors_to_cycle()[5])
    if filename:
        filenamesave = rootsave / filename
        plt.savefig(filenamesave, bbox_inches='tight')
    plt.tight_layout()
    plt.show()


def plot_onlyloss_ripetizioni_stesso_trial_superimposed(xp, dot_key, ylim=None, xlim=None, filename=None, lista_keys=None):
    df = xp.gc.config_dataframe
    multicolumns = tuple(dot_key.split('.'))
    if lista_keys is None:
        try:
            distinte = list(set(xp.diz_trials[dot_key]))
        except TypeError:
            distinte = list(set(map(tuple, xp.diz_trials[dot_key])))
    else:
        distinte = lista_keys
    tot = len(distinte)
    custom_lines = []
    absmin = 100
    absmax = -1
    k = 0
    fig, (ax1,ax2) = plt.subplots(2, 1, figsize=(20, 12))
    for var in distinte:
        colore = get_colors_to_cycle_rainbowN(tot)[k]
        custom_lines.append(Line2D([0], [0], color=colore, lw=3))
        if isinstance(var, tuple):
            xp.gc.list2tuples(df)
        m1 = df[multicolumns] == var
        if m1.sum() > 0:
            res = df[m1]
            for i, (j, row) in enumerate(res.iterrows()):
                testloss = row[('risultati', 'test_loss')]
                acc = row[('risultati', 'test_accuracy')]
                loss_list_min, loss_list_max = min(array_wo_outliers(testloss)), max(array_wo_outliers(testloss))
                absmin = min(absmin, loss_list_min)
                absmax = max(absmax, loss_list_max)
                ax1.plot(testloss, color=colore, alpha=0.5) #, label=var)
                ax2.plot(acc, color=colore, alpha=0.5)
        else:
            ax1.plot(0, color=colore)
        k += 1


    ax1.set_ylabel('Test Loss')
    #plt.xlim(0, 500)
    if ylim:
        ax1.set_ylim(ylim)
    else:
        ax1.set_ylim(absmin, absmax)
    ax2.set_ylim(0,1)
    if xlim:
        ax1.set_xlim(0, xlim)
        ax2.set_xlim(0, xlim)
    ax1.legend(custom_lines, distinte)
    ax2.legend(custom_lines, distinte)

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



def plot_weights_multiple_hist(layers, labels, ax1, absmin, absmax, sequential_colors=False, subplot=236):
    assert len(layers) == len(labels), f"ci sono {len(layers)} layers e {len(labels)} labels"
    #absmin, absmax = np.min([np.min(par) for par in layers]), np.max([np.max(par) for par in layers])
    bins = np.linspace(absmin, absmax, 40)
    alphavalue = 0.7
    #fig, ax1 = plt.subplots()
    fig = plt.gcf()
    #ax1.set(xlim=(0, 10), ylim=(0, 25))

    # primo grafico con l'asse che ho passato da fuori
    i = 0
    axes = [ax1]
    custom_lines = []
    color = get_colors_to_cycle_rainbow8()[i % 8]
    if sequential_colors:
        color = get_colors_to_cycle_sequential(len(layers))[i]
    custom_lines.append(Line2D([0], [0], color=color, lw=3))
    ax1.hist(layers[0], bins, density=True, label=labels[i], stacked=True, alpha=alphavalue, color=color)
    ax1.set_ylim(0,3)
    for i, lay in enumerate(layers[1:]):
        asse = fig.add_subplot(subplot, sharex=ax1, sharey=ax1, label=f"ax{i+1}")
        axes.append(asse)
        color = get_colors_to_cycle_rainbow8()[i+1 % 8]
        if sequential_colors:
            color = get_colors_to_cycle_sequential(len(layers))[i+1]
        asse.hist(lay, bins, density=True, label=labels[i+1], stacked=True, alpha=alphavalue, color=color)
        asse.set_ylim(0,3)
        custom_lines.append(Line2D([0], [0], color=color, lw=3))

    ax1.legend(custom_lines, [f"{e}" for e in labels], loc='upper right')


    xshift=0.02; yshift=0.02
    for i, ax in enumerate(axes[::-1]):
        ax.patch.set_visible(False)
        pos = ax.get_position()
        newpos = Bbox.from_bounds(pos.x0+i*xshift, pos.y0+i*yshift, pos.width, pos.height)
        ax.set_position(newpos)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)

        if i > 0:
            ax.spines["left"].set_visible(False)
            ax.tick_params(labelleft=False, left=False, labelbottom=False)

    #fig.savefig("trial01.pdf", transparent=True)
    #plt.show()

# endregion
